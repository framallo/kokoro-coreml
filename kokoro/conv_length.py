"""Conv1d temporal length helpers matching PyTorch ``nn.Conv1d`` / ``F.conv1d``.

Use these instead of ad-hoc ``L//2`` or ``L*2`` when F0/N curves pass through a
real ``Conv1d`` before aligning with ASR features (see ``istftnet.Decoder``).
"""

from __future__ import annotations

import torch.nn as nn


def conv1d_output_length(
    length_in: int,
    *,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int = 1,
) -> int:
    """Return output sequence length for one Conv1d layer (PyTorch formula).

    ``L_out = (L_in + 2*padding - dilation*(kernel_size - 1) - 1) // stride + 1``
    """
    return (length_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


def conv1d_output_length_from_module(length_in: int, conv: nn.Conv1d) -> int:
    """Return Conv1d output length using ``conv`` kernel/stride/padding/dilation."""
    k = conv.kernel_size[0]
    s = conv.stride[0]
    p = conv.padding[0]
    d = conv.dilation[0]
    return conv1d_output_length(
        length_in, kernel_size=k, stride=s, padding=p, dilation=d
    )


def conv1d_min_input_length_for_output_length(length_out: int, conv: nn.Conv1d) -> int:
    """Smallest ``L_in >= 1`` such that ``conv1d_output_length_from_module(L_in, conv) == length_out``.

    Kokoro's ``F0_conv`` / ``N_conv`` shrink the F0/noise curve time axis; the curve length
    fed to ``Decoder.forward`` must be chosen so that after those convs the time dim matches
    ASR. Replacing ``2 * F`` with this lookup fixes odd ``F`` and arbitrary k/s/p.
    """
    if length_out < 1:
        raise ValueError(f"length_out must be >= 1, got {length_out}")
    # Upper bound: output length grows ~O(L_in / stride); search is cheap for Kokoro sizes.
    s = max(1, int(conv.stride[0]))
    hi = max(length_out * s * 4, length_out + 32)
    for L_in in range(1, hi + 1):
        if conv1d_output_length_from_module(L_in, conv) == length_out:
            return L_in
    raise ValueError(
        f"No input length L_in in [1,{hi}] yields output length {length_out} for conv {conv}"
    )
