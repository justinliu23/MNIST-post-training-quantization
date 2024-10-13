# MNIST-post-training-quantization


This repository contains an implementation of post-training weight quantization for neural networks, focusing on compressing 32-bit floating-point weights to 8-bit integers, significantly reducing model size (~4x) while maintaining minimal accuracy loss. The project was developed as part of Harvard's graduate CS 2420 course (Computing at Scale, Fall 2024).

## Project Overview

Quantization is a crucial optimization technique in neural networks, especially for deploying models on resource-constrained devices. By converting high-precision values (32-bit floats) to lower-precision values (8-bit integers), we can reduce memory consumption and improve inference time without drastically impacting model performance.

### Key Features

- **Quantization**: Maps 32-bit floating-point weights to 8-bit integers using shifting and scaling techniques.
- **Dequantization**: Recovers the original values (with minor errors) from the quantized weights.
- **Performance Testing**: Applied the quantization and dequantization process on the MNIST dataset, evaluating its impact on model accuracy.

## Quantization Process

1. **Shifting**: The weight matrix is shifted such that the minimum value is 0.
2. **Scaling**: The matrix is then scaled to fit within the 0-255 range.
3. **Rounding and Conversion**: The scaled values are rounded and converted to 8-bit integers.

### Example
Given a weight vector `v=[400, 500, 600]`:
- After shifting, `v_shifted = [0, 100, 200]`.
- After scaling, `v_scaled = [0, 127.5, 255]`.
- Final quantized vector, `v_q = [0, 128, 255]`.

## Dequantization Process

The quantized values are scaled and shifted back to recover an approximation of the original weights.

### Example
Given the quantized vector `v_q = [0, 128, 255]`, with the original shift of `400` and a scale of `255/200`:
- Reverse scaling gives `v_unscaled = [0, 100.392, 200]`.
- Reverse shifting results in `v = [400, 500.392, 600]`.

The dequantized values closely approximate the original values with minimal loss of accuracy.

## Setup and Usage

### Requirements

- Python 3.x
- NumPy
- PyTorch
- Matplotlib

