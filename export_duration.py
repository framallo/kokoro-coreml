#!/usr/bin/env python3
"""
Export Kokoro Duration model to CoreML ML Program (.mlpackage) with strict shape bounds
and aliasing fixes to avoid CoreML "tile reps >= 1" and BNNS input/output alias errors.

Inputs:
- checkpoints/config.json
- checkpoints/kokoro-v1_0.pth (optional; will fallback to default constructor)

Outputs:
- coreml/kokoro_duration.mlpackage
"""
import os
from pathlib import Path

import numpy as np
import coremltools as ct

_ROOT = Path(__file__).resolve().parent
import torch
import torch.nn as nn
import torch.nn.functional as F

from kokoro._export_utils import load_kokoro_for_export
from kokoro.coreml_export_verify import (
    assert_no_cpu_fallback_in_logs,
    capture_ane_logs,
    merge_log_checks,
)
from kokoro.coreml_numeric_validate import validate_duration_traced_vs_coreml

kokoro_istftnet, kokoro_modules, kokoro_model = load_kokoro_for_export(suffix="_duration")
KModel = kokoro_model.KModel
AdaLayerNorm = kokoro_modules.AdaLayerNorm

class CoreMLFriendlyTextEncoder(nn.Module):
    def __init__(self, original_encoder):
        super().__init__()
        self.embedding = original_encoder.embedding
        self.cnn = original_encoder.cnn
        self.lstm = MaskedBidirectionalLSTM(original_encoder.lstm)
    def forward(self, x, input_lengths, m):
        valid_mask = (~m).to(dtype=torch.long)
        x = self.embedding(x)
        x = x.transpose(1, 2)
        m = m.unsqueeze(1)
        x.masked_fill_(m, 0.0)
        for c in self.cnn:
            x = c(x)
            x.masked_fill_(m, 0.0)
        x = x.transpose(1, 2)
        x = self.lstm(x, valid_mask)
        x = x.transpose(-1, -2)
        x.masked_fill_(m, 0.0)
        return x

class CoreMLFriendlyDurationEncoder(nn.Module):
    def __init__(self, original_encoder):
        super().__init__()
        self.lstms = nn.ModuleList(
            MaskedBidirectionalLSTM(block) if isinstance(block, nn.LSTM) else block
            for block in original_encoder.lstms
        )
        self.dropout = original_encoder.dropout
    def forward(self, x, style, text_lengths, m):
        masks = m
        valid_mask = (~masks).to(dtype=torch.long)
        x = x.permute(2, 0, 1)
        # Replace expand with explicit repeat operations to avoid tile reps validation issues
        # style is [batch, style_dim], we need [seq_len, batch, style_dim]
        batch_size = x.shape[1]
        seq_len = x.shape[0] 
        style_dim = style.shape[-1]
        s = style.unsqueeze(0).repeat(seq_len, 1, 1)  # [seq_len, batch, style_dim]
        x = torch.cat([x, s], axis=-1)
        x.masked_fill_(masks.unsqueeze(-1).transpose(0, 1), 0.0)
        x = x.transpose(0, 1)
        x = x.transpose(-1, -2)
        for block in self.lstms:
            if isinstance(block, AdaLayerNorm) or type(block).__name__ == "AdaLayerNorm":
                x = block(x.transpose(-1, -2), style).transpose(-1, -2)
                x = torch.cat([x, s.permute(1, 2, 0)], axis=1)
                x.masked_fill_(masks.unsqueeze(-1).transpose(-1, -2), 0.0)
            else:
                x = x.transpose(-1, -2)
                x = block(x, valid_mask)
                x = nn.functional.dropout(x, p=self.dropout, training=False)
                x = x.transpose(-1, -2)
        return x.transpose(-1, -2)

class MaskedBidirectionalLSTM(nn.Module):
    """Exportable one-layer bidirectional LSTM that ignores right-padding.

    The production Kokoro duration path packs variable-length sequences before
    the shared duration LSTM. Static Core ML duration models are right-padded to
    enumerated token counts, so a vanilla bidirectional LSTM would let the
    backward direction read padding before valid tokens. This module reproduces
    packed semantics for trailing-padding masks while remaining traceable for
    each fixed export length.
    """
    def __init__(self, original_lstm: nn.LSTM):
        super().__init__()
        if original_lstm.num_layers != 1 or not original_lstm.bidirectional or not original_lstm.batch_first:
            raise ValueError("MaskedBidirectionalLSTM expects one-layer batch-first bidirectional LSTM")
        self.hidden_size = original_lstm.hidden_size
        self.register_buffer("weight_ih_l0", original_lstm.weight_ih_l0.detach().clone())
        self.register_buffer("weight_hh_l0", original_lstm.weight_hh_l0.detach().clone())
        self.register_buffer("bias_ih_l0", original_lstm.bias_ih_l0.detach().clone())
        self.register_buffer("bias_hh_l0", original_lstm.bias_hh_l0.detach().clone())
        self.register_buffer("weight_ih_l0_reverse", original_lstm.weight_ih_l0_reverse.detach().clone())
        self.register_buffer("weight_hh_l0_reverse", original_lstm.weight_hh_l0_reverse.detach().clone())
        self.register_buffer("bias_ih_l0_reverse", original_lstm.bias_ih_l0_reverse.detach().clone())
        self.register_buffer("bias_hh_l0_reverse", original_lstm.bias_hh_l0_reverse.detach().clone())

    def _cell(
        self,
        x_t: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
        weight_ih: torch.Tensor,
        weight_hh: torch.Tensor,
        bias_ih: torch.Tensor,
        bias_hh: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gates = F.linear(x_t, weight_ih, bias_ih) + F.linear(h, weight_hh, bias_hh)
        i_gate, f_gate, g_gate, o_gate = gates.chunk(4, dim=1)
        i_gate = torch.sigmoid(i_gate)
        f_gate = torch.sigmoid(f_gate)
        g_gate = torch.tanh(g_gate)
        o_gate = torch.sigmoid(o_gate)
        c_new = f_gate * c + i_gate * g_gate
        h_new = o_gate * torch.tanh(c_new)
        return h_new, c_new

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch, steps, _ = x.shape
        mask = attention_mask.to(dtype=x.dtype)
        h_f = x.new_zeros((batch, self.hidden_size))
        c_f = x.new_zeros((batch, self.hidden_size))
        forward_outputs: list[torch.Tensor] = []
        for t in range(steps):
            active = mask[:, t].unsqueeze(1)
            h_new, c_new = self._cell(
                x[:, t, :],
                h_f,
                c_f,
                self.weight_ih_l0,
                self.weight_hh_l0,
                self.bias_ih_l0,
                self.bias_hh_l0,
            )
            h_f = h_new * active + h_f * (1.0 - active)
            c_f = c_new * active + c_f * (1.0 - active)
            forward_outputs.append(h_f * active)

        h_b = x.new_zeros((batch, self.hidden_size))
        c_b = x.new_zeros((batch, self.hidden_size))
        backward_reversed: list[torch.Tensor] = []
        for t in range(steps - 1, -1, -1):
            active = mask[:, t].unsqueeze(1)
            h_new, c_new = self._cell(
                x[:, t, :],
                h_b,
                c_b,
                self.weight_ih_l0_reverse,
                self.weight_hh_l0_reverse,
                self.bias_ih_l0_reverse,
                self.bias_hh_l0_reverse,
            )
            h_b = h_new * active + h_b * (1.0 - active)
            c_b = c_new * active + c_b * (1.0 - active)
            backward_reversed.append(h_b * active)
        backward_outputs = list(reversed(backward_reversed))

        return torch.cat(
            [torch.stack(forward_outputs, dim=1), torch.stack(backward_outputs, dim=1)],
            dim=2,
        )

class DurationModel(nn.Module):
    def __init__(self, kmodel: KModel):
        super().__init__()
        self.kmodel = kmodel
        self.kmodel.text_encoder = CoreMLFriendlyTextEncoder(kmodel.text_encoder)
        self.kmodel.predictor.text_encoder = CoreMLFriendlyDurationEncoder(kmodel.predictor.text_encoder)
        self.duration_lstm = MaskedBidirectionalLSTM(kmodel.predictor.lstm)
        if hasattr(self.kmodel.bert.embeddings, 'token_type_ids'):
            delattr(self.kmodel.bert.embeddings, 'token_type_ids')
    def forward(self, input_ids: torch.LongTensor, ref_s: torch.FloatTensor, speed: torch.FloatTensor, attention_mask: torch.LongTensor):
        k = self.kmodel
        input_lengths = attention_mask.sum(dim=-1).to(torch.long)
        text_mask = attention_mask == 0
        token_type_ids = torch.zeros_like(input_ids)
        bert_dur = k.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        d_en = k.bert_encoder(bert_dur).transpose(-1, -2)
        s = ref_s[:, 128:]  # style half
        d = k.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x = self.duration_lstm(d, attention_mask)
        duration = k.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1).long()
        t_en = k.text_encoder(input_ids, input_lengths, text_mask)
        # Avoid CoreML aliasing: ensure ref_s output is distinct
        ref_s_out = ref_s + torch.zeros_like(ref_s)
        return pred_dur, d, t_en, s, ref_s_out

def remove_training_ops(model):
    """Recursively replace training-specific ops with eval equivalents to avoid TRAINING dialect."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            # Replace dropout with identity
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model
            setattr(parent, child_name, nn.Identity())
        elif isinstance(module, nn.BatchNorm1d):
            # Set to eval mode and freeze
            module.eval()
            module.track_running_stats = False
        elif isinstance(module, nn.LSTM):
            # Ensure LSTM is in eval mode
            module.eval()


def _path_is_readable_file(p: Path) -> bool:
    """True if path is a readable file; False on missing or broken symlinks / permission errors."""
    try:
        return p.is_file()
    except OSError:
        return False


def main():
    cfg = _ROOT / "checkpoints/config.json"
    ckpt = _ROOT / "checkpoints/kokoro-v1_0.pth"
    if _path_is_readable_file(cfg) and _path_is_readable_file(ckpt):
        kmodel = KModel(config=str(cfg), model=str(ckpt), disable_complex=True)
    elif _path_is_readable_file(cfg):
        kmodel = KModel(config=str(cfg), disable_complex=True)
    else:
        kmodel = KModel(disable_complex=True)
    
    duration_model = DurationModel(kmodel)
    # Ensure we're in eval mode and remove training-specific operations
    duration_model.eval()
    remove_training_ops(duration_model)
    
    # Force all submodules to eval mode to prevent TRAINING dialect
    for module in duration_model.modules():
        module.eval()

    # Use torch.export instead of jit.trace to avoid baking shape constants
    # Test with multiple sequence lengths to ensure dynamic shapes work
    test_lengths = [16, 32, 64]
    
    for T in test_lengths:
        input_ids = torch.randint(0, 100, (1, T), dtype=torch.int32)
        ref_s = torch.zeros(1, 256, dtype=torch.float32)
        speed = torch.tensor([1.0], dtype=torch.float32)
        attention_mask = torch.ones(1, T, dtype=torch.int32)
        
        with torch.no_grad():
            outputs = duration_model(input_ids, ref_s, speed, attention_mask)
            print(f"✓ Test T={T}: outputs shapes = {[o.shape for o in outputs]}")

    # E5RT (ANE runtime) cannot handle RangeDim or EnumeratedShapes with multiple
    # variable inputs — it fails with "Tensor size cannot be queried because all
    # dimensions are not known." The workaround is separate models per token count.
    #
    # Export one model per enumerated size. The caller picks the smallest model
    # that fits the actual token count and pads to that size.
    sizes_env = os.environ.get("KOKORO_DURATION_EXPORT_SIZES")
    if sizes_env:
        ENUM_SIZES = [int(value.strip()) for value in sizes_env.split(",") if value.strip()]
    else:
        ENUM_SIZES = [32, 64, 128, 256, 512]
    validate_max_t = int(os.environ.get("KOKORO_DURATION_EXPORT_VALIDATE_MAX_T", "128"))

    out_dir = _ROOT / "coreml"
    out_dir.mkdir(parents=True, exist_ok=True)

    for T in ENUM_SIZES:
        print(f"\n{'='*50}")
        print(f"Exporting Duration model for T={T}")
        print(f"{'='*50}")

        input_ids = torch.randint(0, 100, (1, T), dtype=torch.int32)
        ref_s = torch.zeros(1, 256, dtype=torch.float32)
        speed = torch.tensor([1.0], dtype=torch.float32)
        attention_mask = torch.ones(1, T, dtype=torch.int32)

        with torch.no_grad():
            traced = torch.jit.trace(duration_model, (input_ids, ref_s, speed, attention_mask), strict=False)

        with capture_ane_logs() as convert_buf:
            duration_ml = ct.convert(
                traced,
                inputs=[
                    ct.TensorType(name="input_ids",      shape=(1, T),  dtype=np.int32),
                    ct.TensorType(name="ref_s",          shape=(1, 256),  dtype=np.float32),
                    ct.TensorType(name="speed",          shape=(1,),      dtype=np.float32),
                    ct.TensorType(name="attention_mask", shape=(1, T),  dtype=np.int32),
                ],
            outputs=[
                ct.TensorType(name="pred_dur"),
                ct.TensorType(name="d"), 
                ct.TensorType(name="t_en"),
                ct.TensorType(name="s"),
                ct.TensorType(name="ref_s_out"),  # Renamed to avoid conflict with input
            ],
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.macOS12,
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.ALL,  # Allow ANE optimization
            skip_model_load=T > validate_max_t,
        )
        assert_no_cpu_fallback_in_logs(convert_buf.getvalue(), phase=f"duration T={T} ct.convert")

        out_path = out_dir / f"kokoro_duration_t{T}.mlpackage"
        duration_ml.save(str(out_path))
        print(f"Saved: {out_path}")

        if T <= validate_max_t:
            # Quick validation: predict with the exported model. Larger static
            # packages can spend minutes compiling here; full parity checks run
            # separately after export.
            test_tokens = min(T, T - 2)
            test_ids = np.zeros((1, T), dtype=np.int32)
            test_ids[0, :test_tokens] = np.random.randint(1, 100, test_tokens)
            test_mask = np.zeros((1, T), dtype=np.int32)
            test_mask[0, :test_tokens] = 1

            test_input = {
                "input_ids": test_ids,
                "ref_s": np.zeros((1, 256), dtype=np.float32),
                "speed": np.array([1.0], dtype=np.float32),
                "attention_mask": test_mask,
            }
            test_output = duration_ml.predict(test_input)
            print(f"  Predict OK: output keys = {list(test_output.keys())}")
        else:
            print(f"  Predict skipped for T={T}; validate_max_t={validate_max_t}")

    # Also save the T=128 model as the default (backward compat)
    default_src = out_dir / "kokoro_duration_t128.mlpackage"
    default_dst = out_dir / "kokoro_duration.mlpackage"
    if default_src.exists():
        import shutil
        if default_dst.exists():
            shutil.rmtree(str(default_dst))
        shutil.copytree(str(default_src), str(default_dst))
        print(f"\nCopied {default_src.name} -> {default_dst.name} (backward compat)")

    print(f"\n✅ Duration models exported for T = {ENUM_SIZES}")

if __name__ == "__main__":
    main()
