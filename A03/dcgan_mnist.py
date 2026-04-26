import argparse
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils


@dataclass
class TrainConfig:
    batch_size: int = 128
    latent_dim: int = 100
    epochs: int = 30
    lr: float = 2e-4
    beta1: float = 0.5
    workers: int = 4
    sample_interval: int = 1
    out_dir: str = "outputs"
    seed: int = 42
    use_amp: bool = True
    channels: int = 1


class Generator(nn.Module):
    def __init__(self, latent_dim: int, channels: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, channels: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)


def weights_init(module: nn.Module) -> None:
    classname = module.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)


def ensure_dirs(config: TrainConfig) -> None:
    os.makedirs(config.out_dir, exist_ok=True)
    os.makedirs(os.path.join(config.out_dir, "samples"), exist_ok=True)
    os.makedirs(os.path.join(config.out_dir, "checkpoints"), exist_ok=True)


def get_dataloader(config: TrainConfig) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers,
        pin_memory=True,
        drop_last=True,
    )


def save_grid(images: torch.Tensor, path: str, nrow: int = 10) -> None:
    utils.save_image(images, path, nrow=nrow, normalize=True, value_range=(-1, 1))


def save_loss_curve(g_losses, d_losses, path: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(g_losses, label="Generator Loss")
    plt.plot(d_losses, label="Discriminator Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("DCGAN Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def train(config: TrainConfig) -> None:
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dirs(config)
    dataloader = get_dataloader(config)

    generator = Generator(config.latent_dim, config.channels).to(device)
    discriminator = Discriminator(config.channels).to(device)
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    criterion = nn.BCEWithLogitsLoss()
    optim_g = optim.Adam(generator.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
    optim_d = optim.Adam(discriminator.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
    use_amp = device.type == "cuda" and config.use_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    fixed_noise = torch.randn(100, config.latent_dim, 1, 1, device=device)
    g_losses, d_losses = [], []
    step = 0

    real_label = 1.0
    fake_label = 0.0

    for epoch in range(config.epochs):
        for batch_idx, (real_images, _) in enumerate(dataloader):
            real_images = real_images.to(device, non_blocking=True)
            batch_size = real_images.size(0)
            label_real = torch.full((batch_size,), real_label, device=device)
            label_fake = torch.full((batch_size,), fake_label, device=device)

            discriminator.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                output_real = discriminator(real_images)
                loss_d_real = criterion(output_real, label_real)
            scaler.scale(loss_d_real).backward()

            noise = torch.randn(batch_size, config.latent_dim, 1, 1, device=device)
            fake_images = generator(noise)
            with torch.amp.autocast("cuda", enabled=use_amp):
                output_fake = discriminator(fake_images.detach())
                loss_d_fake = criterion(output_fake, label_fake)
                loss_d = loss_d_real + loss_d_fake
            scaler.scale(loss_d_fake).backward()
            scaler.step(optim_d)

            generator.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                output_gen = discriminator(fake_images)
                loss_g = criterion(output_gen, label_real)
            scaler.scale(loss_g).backward()
            scaler.step(optim_g)
            scaler.update()

            g_losses.append(loss_g.item())
            d_losses.append(loss_d.item())

            if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{config.epochs}] Batch [{batch_idx}/{len(dataloader)}] "
                    f"Loss D: {loss_d.item():.4f}, Loss G: {loss_g.item():.4f}"
                )
            step += 1

        if (epoch + 1) % config.sample_interval == 0:
            with torch.no_grad():
                samples = generator(fixed_noise).detach().cpu()
            sample_path = os.path.join(config.out_dir, "samples", f"epoch_{epoch+1:03d}.png")
            save_grid(samples, sample_path, nrow=10)

            ckpt = {
                "epoch": epoch + 1,
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "optim_g": optim_g.state_dict(),
                "optim_d": optim_d.state_dict(),
                "config": config.__dict__,
            }
            torch.save(ckpt, os.path.join(config.out_dir, "checkpoints", f"dcgan_epoch_{epoch+1:03d}.pt"))

    save_loss_curve(g_losses, d_losses, os.path.join(config.out_dir, "loss_curve.png"))

    with torch.no_grad():
        final_noise = torch.randn(10, config.latent_dim, 1, 1, device=device)
        final_samples = generator(final_noise).detach().cpu()
    save_grid(final_samples, os.path.join(config.out_dir, "generated_10_digits.png"), nrow=10)
    print(f"Training complete. Outputs saved in: {config.out_dir}")


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train DCGAN on MNIST")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sample-interval", type=int, default=1)
    parser.add_argument("--out-dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    return TrainConfig(
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        lr=args.lr,
        beta1=args.beta1,
        workers=args.workers,
        sample_interval=args.sample_interval,
        out_dir=args.out_dir,
        seed=args.seed,
        use_amp=not args.no_amp,
    )


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
