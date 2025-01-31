# main.py
import torch
import logging
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from config import TrainingConfig
from models.generator import Generator
from models.discriminator import Discriminator
from dataset import DATA, get_dataloader
from trainer import ImprovedDeepfakeTrainer
import torch.backends.cudnn
import torch.multiprocessing

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    torch.multiprocessing.set_sharing_strategy('file_system')

    config = TrainingConfig()

    try:
        logger.info("Inicializando dataset...")
        full_dataset = DATA(
            source_dir='/root/dataset/source',
            target_dir='/root/dataset/target',
            image_size=config.image_size
        )

        train_size = int(0.9 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        logger.info(f"Dataset inicializado: {len(full_dataset)} pares")
        logger.info(f"Entrenamiento: {len(train_dataset)} - Validación: {len(val_dataset)}")

        train_loader = get_dataloader(train_dataset, config.batch_size)
        val_loader = get_dataloader(val_dataset, config.batch_size, shuffle=False)

        logger.info("Inicializando modelos...")
        generator = Generator(config).to(config.device)
        discriminator = Discriminator(config).to(config.device)

        # Verificación inicial del generador
        test_input_src = torch.randn(1, 3, 256, 256, device=config.device)  # Imagen fuente de prueba
        test_input_tgt = torch.randn(1, 3, 256, 256, device=config.device)  # Imagen objetivo de prueba
        test_output = generator(test_input_src, test_input_tgt)  # Pasar src y tgt por separado
        logger.info(f"Rango de salida del generador: {test_output.min().item():.2f} - {test_output.max().item():.2f}")

        trainer = ImprovedDeepfakeTrainer(
            config,
            generator,
            discriminator,
            train_loader,
            val_loader,
            logger
        )

        # Verificación de dispositivos de los componentes
        logger.info(f"Dispositivo del generador: {next(generator.parameters()).device}")
        logger.info(f"Dispositivo del discriminador: {next(discriminator.parameters()).device}")
        logger.info(f"Dispositivo de perceptual loss: {next(trainer.perceptual_loss.parameters()).device}")

        logger.info("Comenzando entrenamiento...")
        checkpoint_dir = Path('checkpoints')
        checkpoint_dir.mkdir(exist_ok=True)

        best_model_path = trainer.train(checkpoint_dir)
        logger.info(f"Entrenamiento completado. Mejor modelo guardado en: {best_model_path}")

    except Exception as e:
        logger.error(f"Error en ejecución principal: {str(e)}")
        raise

if __name__ == '__main__':
    main()
