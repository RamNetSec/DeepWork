# models/attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            spectral_norm(nn.Linear(channels, channels // reduction)),
            nn.ReLU(inplace=True),
            spectral_norm(nn.Linear(channels // reduction, channels)),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).reshape(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SelfAttention(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 8):
        super().__init__()
        self.channels = channels
        self.reduction = channels // reduction_ratio

        # Proyecciones
        self.query = nn.utils.spectral_norm(nn.Conv2d(channels, self.reduction, 1))
        self.key = nn.utils.spectral_norm(nn.Conv2d(channels, self.reduction, 1))
        self.value = nn.utils.spectral_norm(nn.Conv2d(channels, channels, 1))

        # Factor de escala corregido
        self.scale = 1.0 / (self.reduction ** 0.5)
        self.gamma = nn.Parameter(torch.zeros(1))

        # Inicialización
        nn.init.xavier_uniform_(self.query.weight, gain=0.1)
        nn.init.xavier_uniform_(self.key.weight, gain=0.1)
        nn.init.xavier_uniform_(self.value.weight, gain=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W

        q = self.query(x).view(B, self.reduction, N)
        k = self.key(x).view(B, self.reduction, N).transpose(1, 2)
        v = self.value(x).view(B, C, N)

        attn_logits = torch.bmm(k, q) * self.scale
        attn_logits = torch.clamp(attn_logits, -50, 50)  # Prevenir overflow
        attn = F.softmax(attn_logits, dim=1)

        out = torch.bmm(v, attn).view(B, C, H, W)
        return x + self.gamma * out


class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.projection = nn.utils.spectral_norm(nn.Conv2d(channels, channels, 1))
        
    def forward(self, x):
        return x + self.projection(x)
