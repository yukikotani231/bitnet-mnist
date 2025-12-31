"""
BitNet vs Standard Linear vs Triton Benchmark
精度と速度の比較
"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from bitnet_triton import BitLinearTriton
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from bitnet_mnist import BitLinear, BitNetMNIST, RMSNorm

# =============================================================================
# 通常のLinearを使ったMNISTモデル（比較用）
# =============================================================================


class StandardMNIST(nn.Module):
    """通常のnn.Linearを使用したMNIST分類器"""

    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(784, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        return self.layers(x)


# =============================================================================
# Triton版BitNet MNISTモデル
# =============================================================================


class BitNetMNISTTriton(nn.Module):
    """BitLinearTritonを使用したMNIST分類器"""

    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.flatten = nn.Flatten()

        self.layers = nn.Sequential(
            BitLinearTriton(784, hidden_dim),
            RMSNorm(hidden_dim),
            nn.GELU(),
            BitLinearTriton(hidden_dim, hidden_dim),
            RMSNorm(hidden_dim),
            nn.GELU(),
            BitLinearTriton(hidden_dim, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        return self.layers(x)

    def pack_weights(self):
        """Pack all BitLinearTriton layers for inference"""
        for module in self.modules():
            if isinstance(module, BitLinearTriton):
                module.pack_weights()

    @classmethod
    def from_bitnet(cls, bitnet_model: BitNetMNIST) -> "BitNetMNISTTriton":
        """Convert BitNetMNIST to BitNetMNISTTriton"""
        hidden_dim = bitnet_model.layers[0].out_features
        triton_model = cls(hidden_dim=hidden_dim)

        # Copy weights from BitLinear to BitLinearTriton
        src_layers = [m for m in bitnet_model.modules() if isinstance(m, BitLinear)]
        dst_layers = [m for m in triton_model.modules() if isinstance(m, BitLinearTriton)]

        for src, dst in zip(src_layers, dst_layers):
            dst.weight.data.copy_(src.weight.data)

        return triton_model


# =============================================================================
# ベンチマーク関数
# =============================================================================


def train_epoch(model, loader, optimizer, device):
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


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total


def benchmark_inference(model, loader, device, num_runs=3):
    """推論速度のベンチマーク"""
    model.eval()

    # ウォームアップ
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            _ = model(images)
            break

    # 計測
    times = []
    for _ in range(num_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(device)
                _ = model(images)

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return sum(times) / len(times)


def benchmark_training(model, loader, optimizer, device, num_epochs=1):
    """学習速度のベンチマーク"""
    model.train()

    # ウォームアップ
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        break

    # 計測
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_epochs):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()

    return time.perf_counter() - start


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # データ
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("./data", train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

    hidden_dim = 512
    epochs = 5
    lr = 1e-3

    # ==========================================================================
    # BitNet モデル（学習）
    # ==========================================================================
    print("=" * 70)
    print("BitNet Model (Training with STE)")
    print("=" * 70)

    bitnet_model = BitNetMNIST(hidden_dim=hidden_dim).to(device)
    bitnet_optimizer = torch.optim.AdamW(bitnet_model.parameters(), lr=lr, weight_decay=0.01)

    print(f"Parameters: {count_parameters(bitnet_model):,}")

    # 学習
    print(f"\nTraining for {epochs} epochs...")
    train_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        loss = train_epoch(bitnet_model, train_loader, bitnet_optimizer, device)
        acc = evaluate(bitnet_model, test_loader, device)
        print(f"  Epoch {epoch}: Loss={loss:.4f}, Acc={acc:.2f}%")

    if device.type == "cuda":
        torch.cuda.synchronize()
    bitnet_train_time = time.perf_counter() - train_start

    bitnet_acc = evaluate(bitnet_model, test_loader, device)
    bitnet_infer_time = benchmark_inference(bitnet_model, test_loader, device)

    # ==========================================================================
    # BitNet Triton モデル（推論）
    # ==========================================================================
    print()
    print("=" * 70)
    print("BitNet Triton Model (Inference with Triton Kernels)")
    print("=" * 70)

    # 学習済みBitNetモデルからTritonモデルへ変換
    triton_model = BitNetMNISTTriton.from_bitnet(bitnet_model).to(device)
    triton_model.pack_weights()  # 重みをパック

    triton_acc = evaluate(triton_model, test_loader, device)
    print(f"Accuracy after packing: {triton_acc:.2f}%")

    triton_infer_time = benchmark_inference(triton_model, test_loader, device)
    print(f"Inference time: {triton_infer_time:.3f}s")

    # ==========================================================================
    # Standard モデル（比較用）
    # ==========================================================================
    print()
    print("=" * 70)
    print("Standard Model (nn.Linear, FP32)")
    print("=" * 70)

    standard_model = StandardMNIST(hidden_dim=hidden_dim).to(device)
    standard_optimizer = torch.optim.AdamW(standard_model.parameters(), lr=lr, weight_decay=0.01)

    print(f"Parameters: {count_parameters(standard_model):,}")

    # 学習
    print(f"\nTraining for {epochs} epochs...")
    train_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        loss = train_epoch(standard_model, train_loader, standard_optimizer, device)
        acc = evaluate(standard_model, test_loader, device)
        print(f"  Epoch {epoch}: Loss={loss:.4f}, Acc={acc:.2f}%")

    if device.type == "cuda":
        torch.cuda.synchronize()
    standard_train_time = time.perf_counter() - train_start

    standard_acc = evaluate(standard_model, test_loader, device)
    standard_infer_time = benchmark_inference(standard_model, test_loader, device)

    # ==========================================================================
    # 結果比較
    # ==========================================================================
    print()
    print("=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    print()
    print(f"{'Metric':<25} {'BitNet STE':>12} {'BitNet Triton':>14} {'Standard':>12}")
    print("-" * 70)
    print(
        f"{'Test Accuracy (%)':<25} {bitnet_acc:>12.2f} {triton_acc:>14.2f} {standard_acc:>12.2f}"
    )
    print(
        f"{'Training Time (s)':<25} {bitnet_train_time:>12.2f} {'N/A':>14} {standard_train_time:>12.2f}"
    )
    print(
        f"{'Inference Time (s)':<25} {bitnet_infer_time:>12.3f} {triton_infer_time:>14.3f} {standard_infer_time:>12.3f}"
    )
    print()

    # スピードアップ計算
    print("Speedup Analysis:")
    print(f"  Triton vs BitNet STE:  {bitnet_infer_time / triton_infer_time:.2f}x")
    print(f"  Triton vs Standard:    {standard_infer_time / triton_infer_time:.2f}x")
    print()

    # メモリ使用量（理論値）
    bitnet_params = count_parameters(bitnet_model)
    standard_params = count_parameters(standard_model)

    fp32_size = bitnet_params * 4 / 1024  # KB
    ternary_size = bitnet_params * 2 / 8 / 1024  # KB (2bit per weight)
    standard_fp32_size = standard_params * 4 / 1024  # KB

    print("Memory Usage (theoretical):")
    print(f"  Standard FP32:     {standard_fp32_size:.1f} KB")
    print(f"  BitNet FP32:       {fp32_size:.1f} KB")
    print(f"  BitNet Packed:     {ternary_size:.1f} KB")
    print(f"  Compression:       {fp32_size / ternary_size:.1f}x")
    print()

    # ==========================================================================
    # 大バッチでのベンチマーク
    # ==========================================================================
    print("=" * 70)
    print("Large Batch Inference Benchmark")
    print("=" * 70)
    print()

    batch_sizes = [32, 64, 128, 256, 512]

    print(f"{'Batch Size':<12} {'Standard (ms)':>14} {'Triton (ms)':>14} {'Speedup':>10}")
    print("-" * 55)

    for bs in batch_sizes:
        test_loader_bs = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=2)

        # Warm up
        with torch.no_grad():
            for images, _ in test_loader_bs:
                _ = standard_model(images.to(device))
                _ = triton_model(images.to(device))
                break

        # Benchmark
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            for images, _ in test_loader_bs:
                _ = standard_model(images.to(device))
        torch.cuda.synchronize()
        std_time = (time.perf_counter() - start) * 1000

        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            for images, _ in test_loader_bs:
                _ = triton_model(images.to(device))
        torch.cuda.synchronize()
        tri_time = (time.perf_counter() - start) * 1000

        speedup = std_time / tri_time
        print(f"{bs:<12} {std_time:>14.2f} {tri_time:>14.2f} {speedup:>10.2f}x")

    print()
    print("NOTE:")
    print("  - BitNet Triton uses 2-bit packed weights (16x memory compression)")
    print("  - Speedup improves with larger batch sizes")
    print("  - For small MNIST model, kernel overhead is significant")


if __name__ == "__main__":
    main()
