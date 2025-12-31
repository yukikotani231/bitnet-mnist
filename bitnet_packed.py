"""
BitNet Packed Inference
2bitパッキングされた重みでの推論実装
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# 2bit Packing Utilities
# =============================================================================


def pack_ternary_weights(weight: torch.Tensor) -> tuple[torch.Tensor, float]:
    """
    重みを{-1, 0, 1}に量子化し、2bitでパッキング

    Args:
        weight: FP32の重み [out_features, in_features]

    Returns:
        packed: パッキングされた重み [out_features, in_features // 16]
        scale: 量子化スケール
    """
    # CPUで処理
    weight = weight.cpu()

    # 量子化
    scale = weight.abs().mean().item()
    w_scaled = weight / max(scale, 1e-5)
    w_ternary = torch.clamp(torch.round(w_scaled), -1, 1).to(torch.int8)

    # {-1, 0, 1} -> {0, 1, 2} にマッピング
    w_mapped = (w_ternary + 1).to(torch.uint8)  # 0, 1, 2

    # 16個の2bit値を1つの32bit整数にパック
    out_features, in_features = weight.shape

    # in_featuresを16の倍数にパディング
    padded_in = ((in_features + 15) // 16) * 16
    if padded_in > in_features:
        w_mapped = F.pad(
            w_mapped, (0, padded_in - in_features), value=1
        )  # 0でパディング（元の値は1=ゼロ）

    # リシェイプしてパック
    w_reshaped = w_mapped.view(out_features, -1, 16)

    # 16個の2bit値を32bitにパック
    packed = torch.zeros(out_features, w_reshaped.shape[1], dtype=torch.int32)
    for i in range(16):
        packed |= w_reshaped[:, :, i].to(torch.int32) << (i * 2)

    return packed, scale


def unpack_ternary_weights(
    packed: torch.Tensor, scale: float, original_in_features: int
) -> torch.Tensor:
    """
    パッキングされた重みを展開

    Args:
        packed: パッキングされた重み [out_features, packed_in_features]
        scale: 量子化スケール
        original_in_features: 元のin_features

    Returns:
        weight: 展開された重み [out_features, in_features]
    """
    out_features, packed_in = packed.shape

    # アンパック
    unpacked = torch.zeros(out_features, packed_in * 16, dtype=torch.float32, device=packed.device)
    for i in range(16):
        # 2bit値を抽出
        vals = ((packed >> (i * 2)) & 0b11).to(torch.float32)
        # {0, 1, 2} -> {-1, 0, 1}
        unpacked[:, i::16] = vals - 1

    # 元のサイズにトリム
    unpacked = unpacked[:, :original_in_features]

    # スケールを適用
    return unpacked * scale


# =============================================================================
# Packed BitLinear Layer (推論専用)
# =============================================================================


class PackedBitLinear(nn.Module):
    """
    2bitパッキングされた重みを使用する推論専用レイヤー
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # パッキングされた重み
        packed_in = (in_features + 15) // 16
        self.register_buffer(
            "packed_weight", torch.zeros(out_features, packed_in, dtype=torch.int32)
        )
        self.register_buffer("scale", torch.tensor(1.0))

    def load_from_bitlinear(self, bitlinear: nn.Module):
        """BitLinearレイヤーから重みをロード"""
        packed, scale = pack_ternary_weights(bitlinear.weight.data)
        self.packed_weight.copy_(packed)
        self.scale.fill_(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 重みを展開（推論時のみ）
        weight = unpack_ternary_weights(self.packed_weight, self.scale.item(), self.in_features).to(
            x.device
        )

        # 入力のスケーリング
        x_scale = x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        x_scaled = x / x_scale

        # 行列演算
        output = F.linear(x_scaled, weight)

        return output * x_scale


# =============================================================================
# Optimized Packed BitLinear (加減算のみ)
# =============================================================================


class OptimizedPackedBitLinear(nn.Module):
    """
    乗算を使わない最適化されたBitLinear
    {-1, 0, 1}の重みなので、加算と減算のみで計算可能
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 正と負の重みをマスクとして保持
        self.register_buffer(
            "positive_mask", torch.zeros(out_features, in_features, dtype=torch.bool)
        )
        self.register_buffer(
            "negative_mask", torch.zeros(out_features, in_features, dtype=torch.bool)
        )
        self.register_buffer("scale", torch.tensor(1.0))

    def load_from_bitlinear(self, bitlinear: nn.Module):
        """BitLinearレイヤーから重みをロード"""
        weight = bitlinear.weight.data.cpu()
        scale = weight.abs().mean().item()
        w_scaled = weight / max(scale, 1e-5)
        w_ternary = torch.clamp(torch.round(w_scaled), -1, 1)

        self.positive_mask.copy_(w_ternary == 1)
        self.negative_mask.copy_(w_ternary == -1)
        self.scale.fill_(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 入力のスケーリング
        x_scale = x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        x_scaled = x / x_scale

        # 乗算なしの行列演算
        # y = W @ x where W ∈ {-1, 0, 1}
        # y[i] = sum(x[j] for j where W[i,j]=1) - sum(x[j] for j where W[i,j]=-1)

        # positive_mask: [out, in], x_scaled: [batch, in]
        # 結果: [batch, out]
        pos_sum = torch.mm(x_scaled, self.positive_mask.float().T)
        neg_sum = torch.mm(x_scaled, self.negative_mask.float().T)

        output = (pos_sum - neg_sum) * self.scale

        return output * x_scale


# =============================================================================
# RMSNorm (inference mode)
# =============================================================================


class RMSNormInference(nn.Module):
    """推論用RMSNorm"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.register_buffer("weight", torch.ones(dim))

    def load_from_rmsnorm(self, rmsnorm: nn.Module):
        self.weight.copy_(rmsnorm.weight.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


# =============================================================================
# Packed MNIST Model
# =============================================================================


class PackedBitNetMNIST(nn.Module):
    """
    パッキングされた重みを使用するMNIST推論モデル
    """

    def __init__(self, hidden_dim: int = 512, optimized: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.optimized = optimized

        LinearClass = OptimizedPackedBitLinear if optimized else PackedBitLinear

        self.flatten = nn.Flatten()

        self.linear1 = LinearClass(784, hidden_dim)
        self.norm1 = RMSNormInference(hidden_dim)

        self.linear2 = LinearClass(hidden_dim, hidden_dim)
        self.norm2 = RMSNormInference(hidden_dim)

        self.linear3 = LinearClass(hidden_dim, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)

        x = self.linear1(x)
        x = self.norm1(x)
        x = F.gelu(x)

        x = self.linear2(x)
        x = self.norm2(x)
        x = F.gelu(x)

        x = self.linear3(x)
        return x

    @classmethod
    def from_trained_model(cls, trained_model: nn.Module, optimized: bool = False):
        """学習済みモデルからパッキングモデルを作成"""

        # hidden_dimを取得
        hidden_dim = trained_model.layers[0].out_features

        packed_model = cls(hidden_dim=hidden_dim, optimized=optimized)

        # 各レイヤーの重みをコピー
        layers = list(trained_model.layers.children())

        packed_model.linear1.load_from_bitlinear(layers[0])  # BitLinear
        packed_model.norm1.load_from_rmsnorm(layers[1])  # RMSNorm

        packed_model.linear2.load_from_bitlinear(layers[3])  # BitLinear
        packed_model.norm2.load_from_rmsnorm(layers[4])  # RMSNorm

        packed_model.linear3.load_from_bitlinear(layers[6])  # BitLinear

        return packed_model


# =============================================================================
# モデルの保存と読み込み
# =============================================================================


def save_packed_model(model: PackedBitNetMNIST, path: str):
    """パッキングされたモデルを保存"""
    state = {
        "hidden_dim": model.hidden_dim,
        "optimized": model.optimized,
        "state_dict": model.state_dict(),
    }
    torch.save(state, path)

    # ファイルサイズを返す
    return Path(path).stat().st_size


def load_packed_model(path: str) -> PackedBitNetMNIST:
    """パッキングされたモデルを読み込み"""
    state = torch.load(path, weights_only=True)
    model = PackedBitNetMNIST(hidden_dim=state["hidden_dim"], optimized=state["optimized"])
    model.load_state_dict(state["state_dict"])
    return model


def get_model_memory_usage(model: nn.Module) -> int:
    """モデルのメモリ使用量（バイト）を計算"""
    total = 0
    for param in model.parameters():
        total += param.numel() * param.element_size()
    for buffer in model.buffers():
        total += buffer.numel() * buffer.element_size()
    return total


# =============================================================================
# テスト
# =============================================================================

if __name__ == "__main__":
    from bitnet_mnist import BitNetMNIST

    print("Testing packing utilities...")

    # テスト用の重み
    weight = torch.randn(64, 128)

    # パック
    packed, scale = pack_ternary_weights(weight)
    print(f"Original shape: {weight.shape}")
    print(f"Packed shape: {packed.shape}")
    print(f"Compression: {weight.numel() * 4} bytes -> {packed.numel() * 4} bytes")

    # アンパック
    unpacked = unpack_ternary_weights(packed, scale, 128)
    print(f"Unpacked shape: {unpacked.shape}")

    # 元の量子化値と比較
    w_quant = torch.clamp(torch.round(weight / scale), -1, 1) * scale
    error = (unpacked - w_quant).abs().max().item()
    print(f"Reconstruction error: {error}")

    print("\nTesting model conversion...")

    # モデル変換テスト
    trained = BitNetMNIST(hidden_dim=256)
    packed_model = PackedBitNetMNIST.from_trained_model(trained, optimized=False)
    optimized_model = PackedBitNetMNIST.from_trained_model(trained, optimized=True)

    # ダミー入力
    x = torch.randn(2, 1, 28, 28)

    with torch.no_grad():
        out_trained = trained(x)
        out_packed = packed_model(x)
        out_optimized = optimized_model(x)

    print(f"Trained output: {out_trained.shape}")
    print(f"Packed output: {out_packed.shape}")
    print(f"Optimized output: {out_optimized.shape}")

    # 出力の差を確認
    diff_packed = (out_trained - out_packed).abs().max().item()
    diff_optimized = (out_trained - out_optimized).abs().max().item()
    print(f"Difference (packed): {diff_packed:.6f}")
    print(f"Difference (optimized): {diff_optimized:.6f}")

    print("\nDone!")
