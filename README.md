# BitNet MNIST

BitNet b1.58 implementation for MNIST classification using PyTorch.

## Overview

This is a minimal implementation of BitNet b1.58 (ternary weight neural network) for testing purposes. The model uses weights quantized to {-1, 0, 1}, achieving ~16x theoretical memory compression compared to FP32.

## Features

- **BitLinear**: Linear layer with ternary weights {-1, 0, 1}
- **STE (Straight-Through Estimator)**: Enables gradient flow through quantization
- **RMSNorm**: Root Mean Square Layer Normalization
- **Per-token activation quantization**: Absmax quantization for activations

## Results

### MNIST Classification
```
Test Accuracy: ~97.5%

Weight distribution (quantized):
  -1: ~33%
   0: ~34%
  +1: ~33%

Memory (theoretical):
  FP32:    2612 KB
  Ternary:  163 KB (6.2%)
```

### BitNet DiT (Diffusion Transformer)

MNIST画像生成のためのDiffusion Transformerも実験。

- `bitnet_diffusion.py`: BitNet版DiT実装
- `bitnet_dit_mnist.pt`: 学習済みモデル（50エポック）
- `samples_*.png`: 生成画像サンプル

### Benchmark Results

GPU（RTX A4000）での速度比較：

| Method | Relative Speed | Memory |
|--------|---------------|--------|
| FP16 + Flash Attention + compile | 1.00x (fastest) | 1x |
| FP16 + Flash Attention | 0.78x | 1x |
| FP32 (naive) | 0.25x | 2x |
| BitNet | 0.09x | 0.0625x (16倍圧縮) |

**結論**: BitNetはGPUでは速度向上しない。メモリ圧縮（16x）が主な利点。

## Usage

```bash
# Install dependencies
uv sync

# Run training
uv run python bitnet_mnist.py
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.0.0
- torchvision >= 0.15.0

## License

MIT License
