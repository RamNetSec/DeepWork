# trainer.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from pathlib import Path
from tqdm import tqdm
import logging
from losses.losses import ImprovedVGGPerceptualLoss
import torchvision


class EMA:
    def __init__(self, model, decay, device):
        self.decay = decay
        self.model = model
        self.shadow = {}
        self.backup = {}
        self.device = device
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().to(self.device)

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        self.backup = {name: param.data for name, param in self.model.named_parameters()}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


class ImprovedDeepfakeTrainer:
    def __init__(self, config, generator, discriminator, train_loader, val_loader, logger):
        self.config = config
        self.logger = logger
        self.generator = generator.to(config.device)
        self.discriminator = discriminator.to(config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = config.device

        self.perceptual_loss = ImprovedVGGPerceptualLoss(device=config.device)
        self.scaler = GradScaler(enabled=config.fp16_training)
        self.writer = SummaryWriter(log_dir='logs')
        self.warmup_steps = config.warmup_steps

        # Optimizadores
        self.optimizer_g = torch.optim.AdamW(
            self.generator.parameters(),
            lr=config.learning_rate_g,
            betas=config.betas,
            weight_decay=config.weight_decay
        )

        self.optimizer_d = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=config.learning_rate_d,
            betas=config.betas,
            weight_decay=config.weight_decay
        )

        # Schedulers
        total_steps = config.num_epochs * len(train_loader)

        def lr_lambda(step, warmup_steps, total_steps, max_lr, min_lr):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress)) + min_lr / max_lr

        self.scheduler_g = LambdaLR(
            self.optimizer_g,
            lr_lambda=lambda step: lr_lambda(
                step,
                config.warmup_steps,
                total_steps,
                config.learning_rate_g,
                config.min_learning_rate
            )
        )

        self.scheduler_d = LambdaLR(
            self.optimizer_d,
            lr_lambda=lambda step: lr_lambda(
                step,
                config.warmup_steps,
                total_steps,
                config.learning_rate_d,
                config.min_learning_rate
            )
        )

        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.global_step = 0
        self.current_epoch = 0

        if config.use_ema:
            self.ema = EMA(self.generator, config.ema_decay, config.device)
        else:
            self.ema = None

    def _compute_gradient_penalty(self, real_samples, fake_samples):
        batch_size = real_samples.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1, device=self.device)
        interpolates = real_samples * alpha + fake_samples * (1 - alpha)
        interpolates.requires_grad_(True)

        d_interpolates = self.discriminator(interpolates)

        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=False,
            retain_graph=False,
            only_inputs=True
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def _forward_generator(self, real_src, real_target):
        with autocast(enabled=self.config.fp16_training):
            fake = self.generator(real_src, real_target)  # Generador recibe src y target

            # Detección de NaN
            if torch.isnan(fake).any():
                self.logger.error("NaN detectado en la salida del generador!")
                raise ValueError("NaN en fake")

            fake = torch.clamp(fake, -1.0, 1.0)
            real_target = real_target.to(self.device)

            d_fake = self.discriminator(fake)
            g_adv = -d_fake.mean()
            g_rec = F.l1_loss(fake, real_target) * self.config.lambda_reconstruction

            # Computar pérdidas perceptuales directamente
            content_loss, style_loss = self.perceptual_loss(fake, real_target)

            g_loss = (
                self.config.lambda_adversarial * g_adv +
                g_rec +
                self.config.lambda_perceptual * content_loss +
                self.config.lambda_style * style_loss
            )

            return fake, g_loss, (g_adv.item(), g_rec.item(), content_loss.item(), style_loss.item())

    def _backward_generator(self, g_loss):
        self.optimizer_g.zero_grad()
        self.scaler.scale(g_loss).backward()

        torch.nn.utils.clip_grad_norm_(
            self.generator.parameters(),
            max_norm=self.config.max_grad_norm,
            norm_type=2.0
        )

        # Actualizar el optimizador con la escala
        self.scaler.step(self.optimizer_g)

    def _forward_discriminator(self, real_src, real_target):
        with autocast(enabled=self.config.fp16_training):
            with torch.no_grad():
                fake = self.generator(real_src, real_target)  # Generador recibe src y target
                fake = torch.clamp(fake, -1.0, 1.0)  # Clamping agregado

            d_real = self.discriminator(real_target)
            d_fake = self.discriminator(fake.detach())

            gp = self._compute_gradient_penalty(real_target, fake)
            d_loss = (
                -torch.mean(d_real) +
                torch.mean(d_fake) +
                self.config.gradient_penalty_weight * gp
            )

            return d_loss

    def _backward_discriminator(self, d_loss):
        self.optimizer_d.zero_grad()
        self.scaler.scale(d_loss).backward()

        torch.nn.utils.clip_grad_norm_(
            self.discriminator.parameters(),
            max_norm=50.0,
            norm_type=2.0
        )

        # Actualizar el optimizador con la escala
        self.scaler.step(self.optimizer_d)

    def _train_batch(self, real_src, real_target):
        real_src = real_src.to(self.device, non_blocking=True)
        real_target = real_target.to(self.device, non_blocking=True)

        try:
            # Entrenar al discriminador
            d_loss = self._forward_discriminator(real_src, real_target)
            self._backward_discriminator(d_loss)

            # Entrenar al generador
            fake, g_loss, loss_components = self._forward_generator(real_src, real_target)
            self._backward_generator(g_loss)

            self.scaler.update()

            if self.config.use_ema and self.global_step % self.config.ema_update_interval == 0:
                self.ema.update()

            return {
                'd_loss': d_loss.item(),
                'g_loss': g_loss.item(),
                'fake': fake.detach(),
                'real': real_target.detach(),
                'src': real_src.detach(),  # Añadido fuente para visualización
                'loss_components': loss_components
            }

        except RuntimeError as e:
            self.logger.error(f"Error en batch: {str(e)}")
            return None

    def _log_metrics(self, metrics, batch_size):
        if metrics is None:
            return

        if self.global_step % 50 == 0:
            self.writer.add_scalars('Loss/Components', {
                'adv': metrics['loss_components'][0],
                'rec': metrics['loss_components'][1],
                'content': metrics['loss_components'][2],
                'style': metrics['loss_components'][3]
            }, self.global_step)

            self.writer.add_scalar('Loss/Generator', metrics['g_loss'], self.global_step)
            self.writer.add_scalar('Loss/Discriminator', metrics['d_loss'], self.global_step)
            self.writer.add_scalar('LR/Generator', self.optimizer_g.param_groups[0]['lr'], self.global_step)
            self.writer.add_scalar('LR/Discriminator', self.optimizer_d.param_groups[0]['lr'], self.global_step)

        if self.global_step % 100 == 0:
            with torch.no_grad():
                # Reconstrucciones y generación de imagen con src en pose de target
                src_recon = self.generator(metrics['src'], metrics['src'])
                target_recon = self.generator(metrics['real'], metrics['real'])
                src_in_target_pose = metrics['fake']

                # Normalizar imágenes a rango [0,1]
                src = (metrics['src'][:4].cpu() + 1) / 2
                src_recon = (src_recon[:4].cpu() + 1) / 2
                target = (metrics['real'][:4].cpu() + 1) / 2
                target_recon = (target_recon[:4].cpu() + 1) / 2
                src_in_target_pose = (src_in_target_pose[:4].cpu() + 1) / 2

                # Crear grid
                grid = torchvision.utils.make_grid(
                    torch.cat([src, src_recon, target, target_recon, src_in_target_pose], dim=0),
                    nrow=5,  # Ajustado para 5 imágenes por fila
                    normalize=False,
                    scale_each=True
                )
                self.writer.add_image('Samples/Src_SrcRecon_Target_TargetRecon_SrcInTargetPose', grid, self.global_step)

    def train_epoch(self):
        self.generator.train()
        self.discriminator.train()

        progress_bar = tqdm(
            self.train_loader,
            desc=f'Epoch {self.current_epoch} [Train]',
            dynamic_ncols=True
        )

        for batch_idx, (real_src, real_target) in enumerate(progress_bar):
            self.global_step += 1

            metrics = self._train_batch(real_src, real_target)

            if metrics is None:
                continue

            progress_bar.set_postfix({
                'D Loss': f"{metrics['d_loss']:.2f}",
                'G Loss': f"{metrics['g_loss']:.2f}",
                'LR G': f"{self.optimizer_g.param_groups[0]['lr']:.2e}",
                'LR D': f"{self.optimizer_d.param_groups[0]['lr']:.2e}"
            })

            self._log_metrics(metrics, real_src.size(0))

    @torch.no_grad()
    def validate(self):
        self.generator.eval()
        total_loss = 0.0
        samples = []

        if self.ema:
            self.ema.apply_shadow()

        progress_bar = tqdm(
            self.val_loader,
            desc=f'Epoch {self.current_epoch} [Val]',
            dynamic_ncols=True
        )

        for real_src, real_target in progress_bar:
            real_src = real_src.to(self.device)
            real_target = real_target.to(self.device)

            fake = self.generator(real_src, real_target)  # Generador recibe src y target
            loss = F.l1_loss(fake, real_target)
            total_loss += loss.item() * real_src.size(0)

            # Nuevo: Extraer y visualizar keypoints
            target_keypoints = self.generator.extract_keypoints(real_target)
            heatmap = target_keypoints.sum(1, keepdim=True).repeat(1, 3, 1, 1)
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)

            if len(samples) < 4:
                samples.extend([
                    (real_src[0] + 1) / 2,        # Source
                    (real_target[0] + 1) / 2,     # Target
                    (fake[0] + 1) / 2,            # Generated
                    heatmap[0].cpu()              # Keypoints Heatmap
                ])

            progress_bar.set_postfix({'Val Loss': f"{loss.item():.2f}"})

        avg_loss = total_loss / len(self.val_loader.dataset)

        # Nuevo grid con visualización de keypoints
        if samples:
            grid = torchvision.utils.make_grid(
                torch.stack(samples),
                nrow=4,  # 4 columnas: source, target, fake, heatmap
                normalize=False
            )
            self.writer.add_image('Validation_Samples', grid, self.current_epoch)

        if self.ema:
            self.ema.restore()

        self.writer.add_scalar('Loss/Validation', avg_loss, self.current_epoch)

        return avg_loss

    def _save_checkpoint(self, checkpoint_dir, is_best=False):
        checkpoint = {
            'epoch': self.current_epoch,
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'optimizer_g': self.optimizer_g.state_dict(),
            'optimizer_d': self.optimizer_d.state_dict(),
            'scaler': self.scaler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': vars(self.config),
            'generator_ema': self.ema.shadow if self.config.use_ema else None,
        }

        filename = 'checkpoint.pt' if not is_best else 'best_model.pt'
        path = checkpoint_dir / filename
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint guardado en {path}")

    def train(self, checkpoint_dir):
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True)

        try:
            for self.current_epoch in range(self.config.num_epochs):
                self.train_epoch()
                val_loss = self.validate()

                self.scheduler_g.step(val_loss)
                self.scheduler_d.step(val_loss)

                self._save_checkpoint(checkpoint_dir)

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.epochs_no_improve = 0
                    self._save_checkpoint(checkpoint_dir, is_best=True)
                else:
                    self.epochs_no_improve += 1

                if self.epochs_no_improve >= self.config.early_stopping_patience:
                    self.logger.info(f"Early stopping en época {self.current_epoch}")
                    break

                self.logger.info(
                    f"Epoch {self.current_epoch} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Best Val Loss: {self.best_val_loss:.4f} | "
                    f"LR G: {self.optimizer_g.param_groups[0]['lr']:.2e} | "
                    f"LR D: {self.optimizer_d.param_groups[0]['lr']:.2e}"
                )

        except KeyboardInterrupt:
            self.logger.info("Entrenamiento interrumpido. Guardando checkpoint...")
            self._save_checkpoint(checkpoint_dir)

        except Exception as e:
            self.logger.error(f"Error crítico: {str(e)}")
            raise e

        finally:
            self.writer.close()
            return checkpoint_dir / 'best_model.pt'
