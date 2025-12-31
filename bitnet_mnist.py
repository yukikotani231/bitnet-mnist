"""
BitNet b1.58 MNIST Implementation
重みを{-1, 0, 1}に量子化したニューラルネットワークでMNISTを学習
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# =============================================================================
# BitLinear: BitNet b1.58 の核心部分
# =============================================================================


class BitLinear(nn.Linear):
    """
    BitNet b1.58 の線形層

    Forward: 重みを{-1, 0, 1}に量子化して計算
    Backward: STE (Straight-Through Estimator) で勾配をそのまま通す
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        # BitNetでは通常biasを使わない
        super().__init__(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 重みを{-1, 0, 1}に量子化
        w_quantized = self.quantize_weights(self.weight)

        # 入力のスケーリング（per-token absmax quantization）
        x_scaled, x_scale = self.activation_quant(x)

        # 量子化された重みで計算
        output = F.linear(x_scaled, w_quantized, self.bias)

        # スケールを戻す
        output = output * x_scale

        return output

    def quantize_weights(self, w: torch.Tensor) -> torch.Tensor:
        """
        重みを{-1, 0, 1}に量子化 (absmean quantization)
        STE: backward時は量子化を無視して勾配を通す
        """
        # スケールファクター: 重みの絶対値平均
        scale = w.abs().mean().clamp(min=1e-5)

        # [-1, 1]にスケーリングして丸め → {-1, 0, 1}
        w_scaled = w / scale
        w_quantized = torch.clamp(torch.round(w_scaled), -1, 1)

        # STE: forwardは量子化値、backwardは元の値の勾配を使う
        return (w_quantized - w).detach() + w

    def activation_quant(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        アクティベーションのper-token absmax量子化
        """
        # 各トークン（バッチの各サンプル）の最大絶対値
        scale = x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        x_scaled = x / scale
        return x_scaled, scale


# =============================================================================
# RMSNorm: BitNetで推奨される正規化層
# =============================================================================


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


# =============================================================================
# BitNet MNIST Model
# =============================================================================


class BitNetMNIST(nn.Module):
    """
    BitNetを使用したMNIST分類器
    """

    def __init__(self, hidden_dim: int = 512):
        super().__init__()

        # MNIST: 28x28 = 784 input features, 10 classes
        self.flatten = nn.Flatten()

        self.layers = nn.Sequential(
            BitLinear(784, hidden_dim),
            RMSNorm(hidden_dim),
            nn.GELU(),
            BitLinear(hidden_dim, hidden_dim),
            RMSNorm(hidden_dim),
            nn.GELU(),
            BitLinear(hidden_dim, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        return self.layers(x)


# =============================================================================
# 学習・評価
# =============================================================================


def train_epoch(
    model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device
) -> float:
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(loader)
    return avg_loss, accuracy


def count_ternary_weights(model: nn.Module) -> dict:
    """量子化後の重みの分布を確認"""
    stats = {"total": 0, "-1": 0, "0": 0, "1": 0}

    for _name, module in model.named_modules():
        if isinstance(module, BitLinear):
            w = module.weight
            scale = w.abs().mean().clamp(min=1e-5)
            w_quantized = torch.clamp(torch.round(w / scale), -1, 1)

            stats["total"] += w_quantized.numel()
            stats["-1"] += int((w_quantized == -1).sum().item())
            stats["0"] += int((w_quantized == 0).sum().item())
            stats["1"] += int((w_quantized == 1).sum().item())

    return stats


def main():
    # 設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    batch_size = 128
    epochs = 10
    lr = 1e-3
    hidden_dim = 512

    # データセット
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("./data", train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # モデル
    model = BitNetMNIST(hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # パラメータ数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Hidden dimension: {hidden_dim}")
    print()

    # 学習
    print("=" * 60)
    print("Training BitNet MNIST")
    print("=" * 60)

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, device)

        print(
            f"Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%"
        )

    # 最終結果
    print()
    print("=" * 60)
    print("Final Results")
    print("=" * 60)

    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"Test Accuracy: {test_acc:.2f}%")

    # 量子化重みの分布
    stats = count_ternary_weights(model)
    print()
    print("Weight distribution (quantized):")
    print(f"  Total weights: {stats['total']:,}")
    print(f"  -1: {stats['-1']:,} ({100 * stats['-1'] / stats['total']:.1f}%)")
    print(f"   0: {stats['0']:,} ({100 * stats['0'] / stats['total']:.1f}%)")
    print(f"  +1: {stats['1']:,} ({100 * stats['1'] / stats['total']:.1f}%)")

    # 理論的なメモリ削減量
    fp32_size = stats["total"] * 4  # 4 bytes per float32
    ternary_size = stats["total"] * 2 / 8  # 2 bits per weight
    print()
    print("Memory (theoretical):")
    print(f"  FP32:    {fp32_size / 1024:.1f} KB")
    print(f"  Ternary: {ternary_size / 1024:.1f} KB ({ternary_size / fp32_size * 100:.1f}%)")


if __name__ == "__main__":
    main()
