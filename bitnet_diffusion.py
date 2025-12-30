"""
BitNet Diffusion Transformer for MNIST
DDPM + DiT-style Transformer with BitLinear layers
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

from bitnet_mnist import BitLinear, RMSNorm


# =============================================================================
# Diffusion Process (DDPM)
# =============================================================================

class GaussianDiffusion:
    """
    DDPM: Denoising Diffusion Probabilistic Models
    """

    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: torch.device = None,
    ):
        self.timesteps = timesteps
        self.device = device or torch.device("cpu")

        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # For q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # For posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        """Forward diffusion: q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]

        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

    @torch.no_grad()
    def p_sample(self, model: nn.Module, x_t: torch.Tensor, t: torch.Tensor):
        """Reverse diffusion: p(x_{t-1} | x_t)"""
        # Predict noise
        predicted_noise = model(x_t, t)

        # Compute x_{t-1}
        beta_t = self.betas[t][:, None, None, None]
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[t][:, None, None, None]

        # Mean of p(x_{t-1} | x_t)
        mean = sqrt_recip_alpha_t * (x_t - beta_t / sqrt_one_minus_alpha_t * predicted_noise)

        if t[0] > 0:
            noise = torch.randn_like(x_t)
            variance = torch.sqrt(self.posterior_variance[t])[:, None, None, None]
            return mean + variance * noise
        else:
            return mean

    @torch.no_grad()
    def sample(self, model: nn.Module, shape: tuple, device: torch.device = None):
        """Generate samples from noise"""
        device = device or self.device
        model.eval()

        # Start from pure noise
        x = torch.randn(shape, device=device)

        for t in tqdm(reversed(range(self.timesteps)), desc="Sampling", total=self.timesteps):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t_batch)

        return x


# =============================================================================
# Time Embedding
# =============================================================================

class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position embedding for timesteps"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


# =============================================================================
# DiT Blocks with BitLinear
# =============================================================================

class BitLinearAttention(nn.Module):
    """Multi-head self-attention with BitLinear"""

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = BitLinear(dim, dim * 3)
        self.proj = BitLinear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class BitLinearMLP(nn.Module):
    """MLP with BitLinear"""

    def __init__(self, dim: int, hidden_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.fc1 = BitLinear(dim, hidden_dim)
        self.fc2 = BitLinear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


class DiTBlock(nn.Module):
    """DiT block with adaptive layer norm"""

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = BitLinearAttention(dim, num_heads)
        self.norm2 = RMSNorm(dim)
        self.mlp = BitLinearMLP(dim)

        # Adaptive layer norm modulation (scale and shift)
        # Use regular Linear - scale/shift need precise values for conditioning
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 4),
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # Get modulation parameters
        mod = self.adaLN(t_emb)
        shift1, scale1, shift2, scale2 = mod.chunk(4, dim=-1)

        # Attention with modulation
        h = self.norm1(x)
        h = h * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        x = x + self.attn(h)

        # MLP with modulation
        h = self.norm2(x)
        h = h * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        x = x + self.mlp(h)

        return x


# =============================================================================
# BitNet DiT Model
# =============================================================================

class BitNetDiT(nn.Module):
    """
    DiT-style diffusion model with BitLinear layers for MNIST
    """

    def __init__(
        self,
        img_size: int = 28,
        patch_size: int = 4,
        in_channels: int = 1,
        dim: int = 256,
        depth: int = 6,
        num_heads: int = 4,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size

        # Patch embedding
        self.patch_embed = nn.Linear(patch_dim, dim)  # Keep as float for input
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, dim) * 0.02)

        # Time embedding (use regular Linear - quantization destroys timestep info)
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(dim, num_heads) for _ in range(depth)
        ])

        # Output
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, patch_dim)  # Keep as float for output

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert image to patches"""
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, -1, C * p * p)
        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert patches back to image"""
        B, N, D = x.shape
        p = self.patch_size
        h = w = int(N ** 0.5)
        c = D // (p * p)
        x = x.reshape(B, h, w, c, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(B, c, h * p, w * p)
        return x

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Patchify
        x = self.patchify(x)
        x = self.patch_embed(x) + self.pos_embed

        # Time embedding
        t_emb = self.time_embed(t)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, t_emb)

        # Output
        x = self.norm(x)
        x = self.head(x)
        x = self.unpatchify(x)

        return x


# =============================================================================
# Training
# =============================================================================

def train_diffusion(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    train_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    device: torch.device = None,
    save_every: int = 10,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for images, _ in pbar:
            images = images.to(device)
            batch_size = images.shape[0]

            # Sample random timesteps
            t = torch.randint(0, diffusion.timesteps, (batch_size,), device=device)

            # Sample noise and create noisy images
            noise = torch.randn_like(images)
            x_t = diffusion.q_sample(images, t, noise)

            # Predict noise
            predicted_noise = model(x_t, t)

            # MSE loss
            loss = F.mse_loss(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")

        # Generate samples
        if epoch % save_every == 0:
            samples = diffusion.sample(model, (16, 1, 28, 28), device)
            samples = (samples.clamp(-1, 1) + 1) / 2  # Normalize to [0, 1]
            save_image(samples, f"samples_epoch_{epoch}.png", nrow=4)
            print(f"Saved samples_epoch_{epoch}.png")

    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)

    # Model
    model = BitNetDiT(
        img_size=28,
        patch_size=4,
        in_channels=1,
        dim=256,
        depth=6,
        num_heads=4,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: BitNet DiT")
    print(f"Parameters: {total_params:,}")
    print(f"Patches: {model.num_patches} (4x4 patches on 28x28)")
    print()

    # Diffusion
    diffusion = GaussianDiffusion(
        timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        device=device,
    )

    # Train
    print("=" * 60)
    print("Training BitNet DiT on MNIST")
    print("=" * 60)

    model = train_diffusion(
        model=model,
        diffusion=diffusion,
        train_loader=train_loader,
        epochs=50,
        lr=1e-3,
        device=device,
        save_every=10,
    )

    # Save model
    torch.save(model.state_dict(), "bitnet_dit_mnist.pt")
    print("Saved model to bitnet_dit_mnist.pt")

    # Generate final samples
    print()
    print("=" * 60)
    print("Generating Final Samples")
    print("=" * 60)

    samples = diffusion.sample(model, (64, 1, 28, 28), device)
    samples = (samples.clamp(-1, 1) + 1) / 2
    save_image(samples, "samples_final.png", nrow=8)
    print("Saved samples_final.png")


if __name__ == "__main__":
    main()
