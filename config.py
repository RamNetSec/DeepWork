# config.py
from dataclasses import dataclass
from typing import Tuple
import torch

@dataclass
class TrainingConfig:
    # Configuración del Modelo
    image_size: int = 256
    channels: int = 3
    num_res_blocks: int = 6  # Experimenta con valores entre 4 y 9
    attention_layers: Tuple[int, ...] = (2, 5)  # Experimenta con diferentes configuraciones

    # Keypoints
    num_keypoints: int = 68
    sigma: float = 0.5  # Nuevo: sigma reducido para heatmaps más definidos
    kernel_size: int = 3  # Nuevo: kernel más pequeño para menos difuminado

    # EMA (Exponential Moving Average)
    use_ema: bool = True
    ema_decay: float = 0.999
    ema_update_interval: int = 1

    # Parámetros de Entrenamiento
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate_g: float = 2e-4
    learning_rate_d: float = 1e-4
    min_learning_rate: float = 1e-6  # Nuevo: LR mínimo
    betas: Tuple[float, float] = (0.5, 0.999)
    weight_decay: float = 1e-5
    gradient_penalty_weight: float = 1.0  # Reducido de 10.0
    warmup_steps: int = 500  # Reducido de 1000
    max_grad_norm: float = 10.0

    # Pesos de Pérdidas (Ajusta estos valores según tus experimentos)
    lambda_reconstruction: float = 1.0  # Experimenta entre 0.5 y 2
    lambda_adversarial: float = 1.0
    lambda_perceptual: float = 0.2  # Aumentado de 0.1
    lambda_style: float = 0.02  # Aumentado de 0.01

    # Sistema
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    fp16_training: bool = False  # Desactivado por ahora, puede causar inestabilidad
    early_stopping_patience: int = 10
