# dataset.py
import numpy as np
import torch
import re
import random
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from typing import Tuple, List
from torch.utils.data import Dataset, DataLoader


class DATA(Dataset):
    def __init__(self, source_dir: str, target_dir: str, image_size: int = 256, augment: bool = True):
        source_path = Path(source_dir)
        target_path = Path(target_dir)

        # Validar existencia de directorios
        if not source_path.exists():
            raise FileNotFoundError(f"Source directory not found: {source_dir}")
        if not target_path.exists():
            raise FileNotFoundError(f"Target directory not found: {target_dir}")

        # Cargar todas las imágenes válidas
        self.source_images = self._load_valid_images(source_path)
        self.target_images = self._load_valid_images(target_path)

        # Emparejar por orden secuencial
        self.pairs = self._create_sequential_pairs()

        print(f"\nDataset creado con {len(self.pairs)} pares")
        print("Ejemplos de pares:")
        for i in range(min(3, len(self.pairs))):
            src = self.pairs[i][0].name
            tgt = self.pairs[i][1].name
            print(f"Par {i}: {src} -> {tgt}")

        # Configurar transformaciones
        self.transform = self._build_transforms(image_size, augment)

    def _load_valid_images(self, path: Path) -> List[Path]:
        """Carga y verifica imágenes válidas"""
        valid_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        image_files = [p for p in path.glob('*') if p.suffix.lower() in valid_extensions]

        # Verificar integridad de las imágenes
        valid_images = []
        for img_path in image_files:
            try:
                with Image.open(img_path) as img:
                    img.verify()  # Verifica la integridad de la imagen
                valid_images.append(img_path)
            except (IOError, SyntaxError):
                print(f"Archivo corrupto o inválido: {img_path}")

        return valid_images

    def _create_sequential_pairs(self) -> List[Tuple[Path, Path]]:
        """Crea pares secuenciales tomando el mínimo de ambas listas"""
        min_length = min(len(self.source_images), len(self.target_images))
        return list(zip(self.source_images[:min_length], self.target_images[:min_length]))

    def _build_transforms(self, image_size: int, augment: bool) -> transforms.Compose:
        """Configura las transformaciones de imagen"""
        transforms_list = [
            transforms.Resize(image_size + 20),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]

        if augment:
            transforms_list.insert(0, transforms.RandomApply([
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
                transforms.RandomGrayscale(p=0.1),
                transforms.RandomAffine(10, translate=(0.1, 0.1), scale=(0.9, 1.1))
            ], p=0.8))

        return transforms.Compose(transforms_list)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        max_retries = 3
        for _ in range(max_retries):
            try:
                src_path, tgt_path = self.pairs[idx]
                source_img = Image.open(src_path).convert('RGB')
                target_img = Image.open(tgt_path).convert('RGB')
                return self.transform(source_img), self.transform(target_img)
            except Exception as e:
                print(f"Error loading pair {idx}: {e}")
                idx = random.randint(0, len(self)-1)  # Intenta con otro índice aleatorio
        return self[0]  # Final fallback: retorna el primer par si todo falla

    def verify_dataset(self):
        """Verificación adicional de integridad del dataset"""
        print("\nVerificando integridad del dataset...")
        for i in range(len(self)):
            try:
                self[i]
            except Exception as e:
                print(f"Error en el par {i}: {e}")
        print("Verificación completada\n")


def get_dataloader(dataset, batch_size, shuffle=True):
    """Crea un DataLoader optimizado con prefetching y múltiples workers."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=1,  # Modificar por el número de núcleos -1; solo si da problemas dejar en 1
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,  # Experimenta con valores entre 2 y 8
        drop_last=True,
        generator=torch.Generator()
    )
