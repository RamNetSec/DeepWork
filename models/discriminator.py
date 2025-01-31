# models/discriminator.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur  # Importar gaussian_blur desde torchvision
from .attention import SelfAttention
import mediapipe as mp
import numpy as np
from torchvision import transforms


class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        def conv_block(in_ch, out_ch, use_attn=False):
            layers = [
                nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 4, 2, 1)),
                nn.GroupNorm(32, out_ch),
                nn.LeakyReLU(0.2, True),
                nn.Dropout2d(0.2)  # Dropout para regularizar
            ]
            if use_attn:
                layers.append(SelfAttention(out_ch))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            conv_block(config.channels, 64),
            conv_block(64, 128, use_attn=True),
            conv_block(128, 256),
            conv_block(256, 512, use_attn=True),
            nn.utils.spectral_norm(nn.Conv2d(512, 1, 4, 1, 1))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, -1.0, 1.0)
        return self.model(x)
