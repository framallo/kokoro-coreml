#!/usr/bin/env python3
"""
Tiny 1D Conv Post-Filter to map Swift HAR iSTFT output -> Golden waveform.

- Architecture: 1D residual stack with dilations, same-length I/O
- Loss: L1 + (1 - correlation) to push waveform shape match
- Input: mono 24 kHz waveform (float32), length fixed to 120000 (5s bucket)
- Output: same shape

Usage:
  Imported by train_postfilter.py
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 9, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.act = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        return x + residual


class TinyPostFilter(nn.Module):
    def __init__(self, hidden_channels: int = 32, num_blocks: int = 8):
        super().__init__()
        self.inp = nn.Conv1d(1, hidden_channels, kernel_size=3, padding=1)
        blocks = []
        dilations = [1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1, 2, 4, 8]
        for d in dilations[:num_blocks]:
            blocks.append(ResidualBlock1D(hidden_channels, kernel_size=9, dilation=d))
        self.blocks = nn.Sequential(*blocks)
        self.out = nn.Conv1d(hidden_channels, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,1,T)
        x = self.inp(x)
        x = self.blocks(x)
        x = self.out(x)
        return x


def correlation_loss(y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # y_*: (B,1,T)
    y_pred = y_pred - y_pred.mean(dim=-1, keepdim=True)
    y_true = y_true - y_true.mean(dim=-1, keepdim=True)
    num = (y_pred * y_true).sum(dim=-1)
    den = torch.sqrt((y_pred * y_pred).sum(dim=-1) * (y_true * y_true).sum(dim=-1) + eps)
    corr = num / (den + eps)
    return 1.0 - corr.mean()


def stft(x: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
    # x: (B,1,T) -> complex STFT (B, F, frames)
    x = x.squeeze(1)
    window = torch.hann_window(n_fft, periodic=True, dtype=x.dtype, device=x.device)
    X = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window, return_complex=True, center=True)
    return X


def multiband_stft_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    # Use a few bands typical for 24 kHz
    cfg = [(256, 64), (512, 128), (1024, 256)]
    total = 0.0
    for n_fft, hop in cfg:
        Yp = stft(y_pred, n_fft, hop)
        Yt = stft(y_true, n_fft, hop)
        mag_p = (Yp.abs() + 1e-6).log()
        mag_t = (Yt.abs() + 1e-6).log()
        total = total + F.l1_loss(mag_p, mag_t)
    return total / len(cfg)


def loss_fn(y_pred: torch.Tensor, y_true: torch.Tensor,
            l1_weight: float = 0.2, corr_weight: float = 0.4, stft_weight: float = 0.4) -> torch.Tensor:
    l1 = F.l1_loss(y_pred, y_true)
    corr = correlation_loss(y_pred, y_true)
    stft_l = multiband_stft_loss(y_pred, y_true)
    return l1_weight * l1 + corr_weight * corr + stft_weight * stft_l
