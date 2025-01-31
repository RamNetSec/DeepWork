# losses/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ImprovedVGGPerceptualLoss(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device

        # Cargar VGG19 con pesos pre-entrenados
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.eval()

        # Selección de capas para perceptual y style loss
        self.style_layers = [1, 6, 11, 20]
        self.content_layers = [3, 8, 15, 22]

        # Crear módulos para las capas seleccionadas
        self.vgg = nn.ModuleList([vgg[i] for i in range(23)])

        # Registro de buffers de normalización
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # Congelar parámetros
        for param in self.parameters():
            param.requires_grad = False

        # Mover todo el módulo al dispositivo correcto
        self.to(self.device)

    def compute_gram(self, x):
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)

    def forward(self, x, y):
        # Normalización robusta con clamping
        x = (x + 1.0) / 2.0  # De [-1,1] a [0,1]
        y = (y.detach() + 1.0) / 2.0

        # Asegurar que los tensores estén en el dispositivo correcto
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std

        # Clamping adicional para estabilidad numérica
        x = torch.clamp(x, -2.5, 2.5)
        y = torch.clamp(y, -2.5, 2.5)

        content_loss = 0.0
        style_loss = 0.0

        # Propagación a través de las capas
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            y = layer(y)

            if i in self.content_layers:
                content_loss += F.l1_loss(x, y)
            if i in self.style_layers:
                style_loss += F.l1_loss(self.compute_gram(x), self.compute_gram(y))

        return content_loss, style_loss

__all__ = ['ImprovedVGGPerceptualLoss']
