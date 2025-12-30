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
