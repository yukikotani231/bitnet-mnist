"""
BitNet Full Pipeline Demo
学習 → 2bitパッキング → 推論のフルパイプライン
"""

import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from bitnet_mnist import BitLinear, BitNetMNIST
from bitnet_packed import (
    PackedBitNetMNIST,
    get_model_memory_usage,
    load_packed_model,
    save_packed_model,
)


def train_model(model, train_loader, device, epochs=5, lr=1e-3):
    """モデルを学習"""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    for epoch in range(1, epochs + 1):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch}/{epochs}: Loss = {avg_loss:.4f}")

    return model


def evaluate(model, test_loader, device):
    """精度を評価"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return 100.0 * correct / total


def benchmark_inference(model, test_loader, device, num_runs=5):
    """推論速度をベンチマーク"""
    model.eval()

    # ウォームアップ
    with torch.no_grad():
        for images, _ in test_loader:
            _ = model(images.to(device))
            break

    # 計測
    times = []
    for _ in range(num_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        with torch.no_grad():
            for images, _ in test_loader:
                _ = model(images.to(device))

        if device.type == "cuda":
            torch.cuda.synchronize()

        times.append(time.perf_counter() - start)

    return sum(times) / len(times)


def main():
    print("=" * 70)
    print("BitNet Full Pipeline Demo")
    print("=" * 70)
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # データ準備
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("./data", train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

    output_dir = Path("./models")
    output_dir.mkdir(exist_ok=True)

    # =========================================================================
    # Phase 1: 学習
    # =========================================================================
    print()
    print("=" * 70)
    print("Phase 1: Training BitNet Model")
    print("=" * 70)

    hidden_dim = 512
    model = BitNetMNIST(hidden_dim=hidden_dim).to(device)
    print(f"Hidden dim: {hidden_dim}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    print("Training...")
    train_model(model, train_loader, device, epochs=5)

    train_acc = evaluate(model, test_loader, device)
    print(f"\nTrained model accuracy: {train_acc:.2f}%")

    # FP32モデルを保存
    fp32_path = output_dir / "bitnet_fp32.pt"
    torch.save(model.state_dict(), fp32_path)
    fp32_size = fp32_path.stat().st_size

    # =========================================================================
    # Phase 2: 2bitパッキング（エクスポート）
    # =========================================================================
    print()
    print("=" * 70)
    print("Phase 2: Export to Packed Format")
    print("=" * 70)

    # 通常のパッキングモデル
    packed_model = PackedBitNetMNIST.from_trained_model(model, optimized=False).to(device)
    packed_path = output_dir / "bitnet_packed.pt"
    packed_size = save_packed_model(packed_model, packed_path)

    # 最適化モデル（乗算なし）
    optimized_model = PackedBitNetMNIST.from_trained_model(model, optimized=True).to(device)
    optimized_path = output_dir / "bitnet_optimized.pt"
    optimized_size = save_packed_model(optimized_model, optimized_path)

    print(f"FP32 model size:      {fp32_size:,} bytes ({fp32_size / 1024:.1f} KB)")
    print(f"Packed model size:    {packed_size:,} bytes ({packed_size / 1024:.1f} KB)")
    print(f"Optimized model size: {optimized_size:,} bytes ({optimized_size / 1024:.1f} KB)")
    print(f"Compression ratio:    {fp32_size / packed_size:.1f}x")

    # =========================================================================
    # Phase 3: 推論ベンチマーク
    # =========================================================================
    print()
    print("=" * 70)
    print("Phase 3: Inference Benchmark")
    print("=" * 70)

    # モデルをロード
    packed_loaded = load_packed_model(packed_path).to(device)
    optimized_loaded = load_packed_model(optimized_path).to(device)

    # 精度確認
    packed_acc = evaluate(packed_loaded, test_loader, device)
    optimized_acc = evaluate(optimized_loaded, test_loader, device)

    print("\nAccuracy comparison:")
    print(f"  Original (FP32 weights): {train_acc:.2f}%")
    print(f"  Packed (2bit):           {packed_acc:.2f}%")
    print(f"  Optimized (no multiply): {optimized_acc:.2f}%")

    # 速度ベンチマーク
    print("\nInference speed (5 runs average):")

    original_time = benchmark_inference(model, test_loader, device)
    packed_time = benchmark_inference(packed_loaded, test_loader, device)
    optimized_time = benchmark_inference(optimized_loaded, test_loader, device)

    print(f"  Original:  {original_time:.3f}s")
    print(f"  Packed:    {packed_time:.3f}s ({original_time / packed_time:.2f}x)")
    print(f"  Optimized: {optimized_time:.3f}s ({original_time / optimized_time:.2f}x)")

    # メモリ使用量
    print("\nRuntime memory usage:")
    original_mem = get_model_memory_usage(model)
    packed_mem = get_model_memory_usage(packed_loaded)
    optimized_mem = get_model_memory_usage(optimized_loaded)

    print(f"  Original:  {original_mem:,} bytes ({original_mem / 1024:.1f} KB)")
    print(
        f"  Packed:    {packed_mem:,} bytes ({packed_mem / 1024:.1f} KB) - {original_mem / packed_mem:.1f}x smaller"
    )
    print(f"  Optimized: {optimized_mem:,} bytes ({optimized_mem / 1024:.1f} KB)")

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Metric':<25} {'Original':>12} {'Packed':>12} {'Optimized':>12}")
    print("-" * 70)
    print(f"{'Accuracy (%)':<25} {train_acc:>12.2f} {packed_acc:>12.2f} {optimized_acc:>12.2f}")
    print(
        f"{'File Size (KB)':<25} {fp32_size / 1024:>12.1f} {packed_size / 1024:>12.1f} {optimized_size / 1024:>12.1f}"
    )
    print(
        f"{'Inference Time (s)':<25} {original_time:>12.3f} {packed_time:>12.3f} {optimized_time:>12.3f}"
    )
    print(
        f"{'Memory (KB)':<25} {original_mem / 1024:>12.1f} {packed_mem / 1024:>12.1f} {optimized_mem / 1024:>12.1f}"
    )
    print()

    print("Notes:")
    print("  - Packed model stores weights in 2-bit format (16x compression on disk)")
    print("  - Optimized model uses addition/subtraction only (no multiplication)")
    print("  - Real speedup requires custom CUDA kernels (like bitnet.cpp)")
    print("  - Current implementation unpacks weights at runtime for compatibility")
    print()

    # 重みの分布を表示
    print("Weight distribution (original model):")
    total = 0
    neg_count = 0
    zero_count = 0
    pos_count = 0

    for _name, module in model.named_modules():
        if isinstance(module, BitLinear):
            w = module.weight.data
            scale = w.abs().mean()
            w_q = torch.clamp(torch.round(w / scale), -1, 1)
            total += w_q.numel()
            neg_count += (w_q == -1).sum().item()
            zero_count += (w_q == 0).sum().item()
            pos_count += (w_q == 1).sum().item()

    print(f"  -1: {neg_count:,} ({100 * neg_count / total:.1f}%)")
    print(f"   0: {zero_count:,} ({100 * zero_count / total:.1f}%)")
    print(f"  +1: {pos_count:,} ({100 * pos_count / total:.1f}%)")


if __name__ == "__main__":
    main()
