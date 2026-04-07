"""Tests for ``kokoro.conv_length`` inverse helper (F0_conv length contract)."""

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from kokoro.conv_length import (
    conv1d_min_input_length_for_output_length,
    conv1d_output_length_from_module,
)


def test_min_input_round_trip_matches_f0_conv():
    """Kokoro Decoder F0_conv: k=3, s=2, p=1 (see istftnet.Decoder)."""
    conv = nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1)
    for length_out in range(1, 33):
        L_in = conv1d_min_input_length_for_output_length(length_out, conv)
        assert conv1d_output_length_from_module(L_in, conv) == length_out
