# models/generator.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur
import mediapipe as mp
import numpy as np
from torchvision import transforms
from .attention import ChannelAttention


class ResBlock(nn.Module):
    def __init__(self, channels: int, use_attention: bool = False):
        super().__init__()
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(channels, channels, 3, 1, 1))
        self.norm1 = nn.InstanceNorm2d(channels)
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(channels, channels, 3, 1, 1))
        self.norm2 = nn.InstanceNorm2d(channels)
        self.attention = ChannelAttention(channels) if use_attention else None
        self.alpha = nn.Parameter(torch.tensor(0.1))  # Conexión de salto aprendible

        # Inicialización estable
        nn.init.xavier_uniform_(self.conv1.weight, gain=0.1)
        nn.init.xavier_uniform_(self.conv2.weight, gain=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.leaky_relu(self.norm1(self.conv1(x)), 0.2)
        x = self.norm2(self.conv2(x))
        if self.attention:
            x = self.attention(x)
        return residual + self.alpha * x


class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Encoder
        self.encoder = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(3, 64, 7, 1, 3)),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.utils.spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.utils.spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, True),
        )

        # Bloques residuales
        self.res_blocks = nn.Sequential(*[
            ResBlock(256 + config.num_keypoints, use_attention=(i in config.attention_layers))
            for i in range(config.num_res_blocks)
        ])

        # Decoder
        self.decoder = nn.Sequential(
            nn.utils.spectral_norm(nn.ConvTranspose2d(256 + config.num_keypoints, 128, 4, 2, 1)),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.utils.spectral_norm(nn.ConvTranspose2d(128, 64, 4, 2, 1)),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.utils.spectral_norm(nn.Conv2d(64, 3, 7, 1, 3)),
            nn.Tanh()  # Activación final corregida
        )

        # Inicialización de pesos
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=0.2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Módulo de MediaPipe para la detección de keypoints
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        self.num_keypoints = config.num_keypoints  # Número de keypoints

        # Parámetros para Gaussian Blur
        self.kernel_size = getattr(config, 'kernel_size', 3)  # Define kernel_size, valor predeterminado=3
        self.sigma = getattr(config, 'sigma', 0.5)          # Define sigma, valor predeterminado=0.5

    def extract_keypoints(self, images):
        """
        Extrae keypoints faciales usando MediaPipe para cada imagen en el batch.

        Args:
            images: Tensor de imágenes [B, C, H, W] en rango [-1, 1].

        Returns:
            Tensor de keypoints [B, num_keypoints, H, W] como mapa de calor.
        """
        batch_size, _, H, W = images.shape
        heatmaps_list = []

        # Denormalizar correctamente [-1, 1] -> [0, 1]
        images_denorm = (images * 0.5) + 0.5

        for i in range(batch_size):
            # Convertir a formato PIL correctamente
            image = (images_denorm[i] * 255).clamp(0, 255).to(torch.uint8)
            image_pil = transforms.ToPILImage()(image.cpu())
            image_np = np.array(image_pil)

            results = self.mp_face_mesh.process(image_np)
            heatmaps = torch.zeros((self.num_keypoints, H, W), dtype=torch.float32, device=images.device)

            if results.multi_face_landmarks:
                for j, landmark in enumerate(results.multi_face_landmarks[0].landmark):
                    if j < self.num_keypoints:
                        x = int(landmark.x * W)
                        y = int(landmark.y * H)
                        if 0 <= x < W and 0 <= y < H:
                            heatmaps[j, y, x] = 1.0
                            heatmaps[j] = gaussian_blur(
                                heatmaps[j].unsqueeze(0),
                                kernel_size=self.kernel_size,
                                sigma=self.sigma
                            ).squeeze(0)

            heatmaps_list.append(heatmaps)

        heatmaps_tensor = torch.stack(heatmaps_list, dim=0)  # [B, num_keypoints, H, W]
        return heatmaps_tensor

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)  # [B, 256, 64, 64]

        # Extraer keypoints de target
        target_keypoints = self.extract_keypoints(target)  # [B, num_keypoints, 256, 256]

        # Redimensionar keypoints para que coincidan con la salida del encoder
        target_keypoints = F.interpolate(target_keypoints, size=x.shape[2:], mode='bilinear', align_corners=False)  # [B, num_keypoints, 64, 64]

        # Concatenar keypoints a la salida del encoder
        x = torch.cat([x, target_keypoints], dim=1)  # [B, 256 + num_keypoints, 64, 64]

        x = self.res_blocks(x)  # [B, 256 + num_keypoints, 64, 64]
        x = self.decoder(x)  # [B, 3, 256, 256]
        return x
