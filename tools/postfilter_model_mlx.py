#!/usr/bin/env python3
"""
MLX port of the Tiny 1D Conv Post-Filter.
"""
from __future__ import annotations
import mlx.core as mx
import mlx.nn as nn
import numpy as np


class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 9, dilation: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, stride=1,
                               padding=(kernel_size - 1) // 2 * dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, stride=1,
                               padding=(kernel_size - 1) // 2 * dilation, dilation=dilation)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, T, C)
        residual = x
        x = nn.silu(self.conv1(x))
        x = self.conv2(x)
        return x + residual


class TinyPostFilter(nn.Module):
    def __init__(self, hidden_channels: int = 32, num_blocks: int = 8):
        super().__init__()
        # channels-last: (B, T, C)
        self.inp = nn.Conv1d(in_channels=1, out_channels=hidden_channels, kernel_size=3, padding=1)
        
        blocks = []
        dilations = [1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1, 2, 4, 8]
        for d in dilations[:num_blocks]:
            blocks.append(ResidualBlock1D(hidden_channels, kernel_size=9, dilation=d))
        self.blocks = nn.Sequential(*blocks)
        
        self.out = nn.Conv1d(in_channels=hidden_channels, out_channels=1, kernel_size=3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, T, 1)
        x = self.inp(x)
        x = self.blocks(x)
        x = self.out(x)
        x = mx.tanh(x)
        return x  # (B, T, 1)


def correlation_loss(y_pred: mx.array, y_true: mx.array, eps: float = 1e-8) -> mx.array:
    # y_*: (B, T, 1) -> operate on (B, T)
    y_pred_t = y_pred.squeeze(-1)
    y_true_t = y_true.squeeze(-1)
    y_pred_t = y_pred_t - y_pred_t.mean(axis=-1, keepdims=True)
    y_true_t = y_true_t - y_true_t.mean(axis=-1, keepdims=True)
    num = (y_pred_t * y_true_t).sum(axis=-1)
    den = mx.sqrt((y_pred_t * y_pred_t).sum(axis=-1) * (y_true_t * y_true_t).sum(axis=-1) + eps)
    corr = num / (den + eps)
    return (1.0 - corr).mean()


def stft(x: mx.array, n_fft: int, hop_length: int) -> mx.array:
    # x: (B, T) -> complex STFT (B, F, frames)
    B, T = x.shape
    window = mx.array(np.hanning(n_fft), dtype=mx.float32)
    pad_amount = n_fft // 2
    x_padded = mx.pad(x, [(0, 0), (pad_amount, pad_amount)])
    num_frames = (x_padded.shape[1] - n_fft) // hop_length + 1
    frames = []
    for i in range(int(num_frames)):
        start = i * hop_length
        frames.append(x_padded[:, start:start + n_fft])
    frames = mx.stack(frames, axis=1)  # (B, frames, n_fft)
    frames = frames * window
    X = mx.fft.rfft(frames, n=n_fft, axis=-1)  # (B, frames, F)
    return X.transpose(0, 2, 1)


def multiband_stft_loss(y_pred: mx.array, y_true: mx.array) -> mx.array:
    cfg = [(256, 64), (512, 128), (1024, 256)]
    total = mx.array(0.0)
    
    y_pred_t = y_pred.squeeze(-1)  # (B, T)
    y_true_t = y_true.squeeze(-1)  # (B, T)

    for n_fft, hop in cfg:
        Yp = stft(y_pred_t, n_fft, hop)
        Yt = stft(y_true_t, n_fft, hop)
        mag_p = mx.log(mx.abs(Yp) + 1e-6)
        mag_t = mx.log(mx.abs(Yt) + 1e-6)
        total = total + mx.mean(mx.abs(mag_p - mag_t))
    return total / len(cfg)


def loss_fn(y_pred: mx.array, y_true: mx.array,
            l1_weight: float = 0.2, corr_weight: float = 0.4, stft_weight: float = 0.4) -> mx.array:
    l1 = mx.mean(mx.abs(y_pred - y_true))
    corr = correlation_loss(y_pred, y_true)
    if stft_weight > 0.0:
        stft_l = multiband_stft_loss(y_pred, y_true)
    else:
        stft_l = mx.array(0.0)
    return l1_weight * l1 + corr_weight * corr + stft_weight * stft_l
