# Engineering Plan: Fix CoreML BNNS Crash in Kokoro TTS Export

## Executive Summary
Fix a critical CoreML runtime crash (`EXC_BAD_ACCESS` in BNNS LSTM kernel) caused by improper tensor initialization during model export. The root cause is tensors being created dynamically in `forward()` methods instead of being registered in `__init__()`, causing torch.jit.trace to produce malformed graphs.

## Problem Statement
- **Error**: `EXC_BAD_ACCESS (code=2, address=0x16bc43ff8)` in `libBNNS.dylib`
- **Location**: LSTM operations in synthesizer model during CoreML inference
- **Root Cause**: Tensors created in `forward()` methods aren't properly registered with PyTorch's module system, causing CoreML's converter to generate incorrect memory layouts

## Implementation Plan

### Phase 1: Setup (30 minutes)
1. **Create Fix Branch** (Complete)
   ```bash
   git checkout <commit-with-export-script>
   git checkout -b fix/coreml-bnns-crash
   ```

2. **Verify Baseline**
   - Run existing export script to confirm it still produces the crash
   - Save the crashing .mlpackage for comparison

### Phase 2: Code Changes (2-3 hours)

#### Fix 1: DurationModel Tensor Registration
**File**: `export_synthesizers.py` (or separate module file if refactored)

**Current Problem Code**:
```python
def forward(self, input_ids, ref_s, speed, attention_mask):
    token_type_ids = torch.zeros_like(input_ids)  # BAD: Created in forward
    # ...
    ref_s_out = ref_s + torch.zeros_like(ref_s)   # BAD: Dynamic creation
```

**Fixed Code**:
```python
class DurationModel(nn.Module):
    def __init__(self, kmodel: KModel):
        super().__init__()
        
        # Register all buffers FIRST
        self.register_buffer('token_type_template', torch.zeros(1, 1, dtype=torch.long))
        self.register_buffer('zeros_template', torch.zeros(1, 1))
        
        # Then do ALL model surgery
        kmodel.text_encoder = CoreMLFriendlyTextEncoder(kmodel.text_encoder)
        kmodel.predictor.text_encoder = CoreMLFriendlyDurationEncoder(kmodel.predictor.text_encoder)
        if hasattr(kmodel.bert.embeddings, 'token_type_ids'):
            delattr(kmodel.bert.embeddings, 'token_type_ids')
        
        # Finally assign the modified model
        self.kmodel = kmodel
    
    def forward(self, input_ids, ref_s, speed, attention_mask):
        k = self.kmodel
        
        # Use registered buffers, expand as needed
        token_type_ids = self.token_type_template.expand_as(input_ids).contiguous()
        
        input_lengths = attention_mask.sum(dim=-1).to(torch.long)
        text_mask = attention_mask == 0
        
        bert_dur = k.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        d_en = k.bert_encoder(bert_dur).transpose(-1, -2)
        s = ref_s[:, CoreMLExportConstants.VOICE_STYLE_DIM:]
        
        d = k.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = k.predictor.lstm(d)
        duration = k.predictor.duration_proj(x)
        
        duration = torch.sigmoid(duration).sum(axis=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1).long()
        
        t_en = k.text_encoder(input_ids, input_lengths, text_mask)
        
        # Use registered buffer for ref_s copy
        ref_s_out = ref_s + self.zeros_template.expand_as(ref_s).contiguous()
        
        return pred_dur, d, t_en, s, ref_s_out
```

#### Fix 2: SynthesizerModel Tensor Registration
**Current Problem Code**:
```python
def forward(self, d, t_en, s, ref_s, pred_aln_trg):
    # ...
    F0_pred = en.new_zeros((B, F * 2))  # BAD: Created in forward
    N_pred = en.new_zeros((B, F * 2))    # BAD: Created in forward
    # ...
    if t_en.shape[1] != expected_in:
        pad_ch = expected_in - t_en.shape[1]
        # BAD: Dynamic tensor creation for padding
        t_en = torch.cat([t_en, t_en.new_zeros((t_en.shape[0], pad_ch, t_en.shape[2]))], dim=1)
```

**Fixed Code**:
```python
class SynthesizerModel(nn.Module):
    def __init__(self, kmodel: KModel):
        super().__init__()
        
        # Calculate expected channels FIRST
        expected_in = kmodel.decoder.encode.conv1.in_channels - 2
        
        # Register ALL buffers
        self.register_buffer('zeros_F0', torch.zeros(1, 1))
        self.register_buffer('zeros_N', torch.zeros(1, 1))
        self.register_buffer('pad_template', torch.zeros(1, 1, 1))
        
        # Store expected channels as constant
        self.expected_in = expected_in
        
        # Do ALL model surgery
        kmodel.text_encoder = CoreMLFriendlyTextEncoder(kmodel.text_encoder)
        
        # Finally assign
        self.kmodel = kmodel
    
    def forward(self, d, t_en, s, ref_s, pred_aln_trg):
        k = self.kmodel
        
        # Align temporal lengths
        if t_en.shape[-1] != d.shape[-1]:
            t_en = torch.nn.functional.interpolate(t_en, size=d.shape[-1], mode='nearest')
        
        # Matrix multiplication without einsum
        B = d.shape[0]
        pred_bt = pred_aln_trg.transpose(0, 1).unsqueeze(0).expand(B, -1, -1)
        d_bt = d.transpose(1, 2)
        en = torch.bmm(pred_bt, d_bt)
        
        B, F, H = en.shape
        
        # Use registered buffers for F0/N predictions
        F0_pred = self.zeros_F0.expand(B, F * 2).contiguous()
        N_pred = self.zeros_N.expand(B, F * 2).contiguous()
        
        # Handle channel alignment with registered buffer
        if t_en.shape[1] != self.expected_in:
            if t_en.shape[1] > self.expected_in:
                t_en = t_en[:, :self.expected_in, :]
            else:
                pad_ch = self.expected_in - t_en.shape[1]
                pad = self.pad_template.expand(t_en.shape[0], pad_ch, t_en.shape[2]).contiguous()
                t_en = torch.cat([t_en, pad], dim=1)
        
        # Align text features to frames
        pred_btf = pred_aln_trg.unsqueeze(0).expand(B, -1, -1)
        asr = torch.bmm(t_en, pred_btf)
        
        audio = k.decoder(asr, F0_pred, N_pred, ref_s[:, :CoreMLExportConstants.VOICE_BASELINE_DIM]).squeeze(0)
        return audio
```

#### Fix 3: Add Validation Function
Add this before export to verify all tensors are registered:

```python
def validate_module_buffers(module, module_name=""):
    """Validate that a module has no dynamic tensor creation in forward()"""
    issues = []
    
    # Check for common problematic patterns in source code
    import inspect
    try:
        source = inspect.getsource(module.forward)
        problematic_patterns = [
            'torch.zeros(',
            'torch.ones(',
            'torch.randn(',
            '.new_zeros(',
            '.new_ones(',
            'zeros_like(',
            'ones_like(',
            'torch.tensor(',
        ]
        
        for pattern in problematic_patterns:
            if pattern in source and 'self.' not in source.split(pattern)[0][-20:]:
                issues.append(f"{module_name}: Found '{pattern}' in forward() - should be registered buffer")
    except:
        pass  # Some modules might not have accessible source
    
    # Recursively check submodules
    for name, submodule in module.named_children():
        child_name = f"{module_name}.{name}" if module_name else name
        issues.extend(validate_module_buffers(submodule, child_name))
    
    return issues
```

### Phase 3: Testing Protocol (2 hours)

#### Step 1: Pre-Export Validation
```python
# In export_synthesizers() function, add before tracing:
print("Validating tensor registration...")
issues = validate_module_buffers(synthesizer_model_base, "SynthesizerModel")
if issues:
    print("WARNING: Found potential tensor registration issues:")
    for issue in issues:
        print(f"  - {issue}")
```

#### Step 2: Export Test
```bash
# Run with minimal configuration first
python export_synthesizers.py --buckets="3s" --debug
```

#### Step 3: CoreML Inference Test
```python
import coremltools as ct
import numpy as np

# Load the fixed model
model = ct.models.MLModel("coreml/kokoro_synthesizer_3s.mlpackage")

# Create test inputs matching the expected shapes
inputs = {
    "d": np.random.randn(1, 512, 256).astype(np.float32),
    "t_en": np.random.randn(1, 768, 256).astype(np.float32),
    "s": np.random.randn(1, 128).astype(np.float32),
    "ref_s": np.random.randn(1, 256).astype(np.float32),
    "pred_aln_trg": np.random.randn(256, 1280).astype(np.float32)
}

# This should NOT crash with BNNS error
try:
    output = model.predict(inputs)
    print("✅ Model inference successful!")
except Exception as e:
    print(f"❌ Model inference failed: {e}")
```

#### Step 4: Swift Integration Test
Test with the actual Swift app to ensure the fix works in production.

### Phase 4: Verification Checklist

- [ ] No `torch.zeros()`, `torch.ones()`, or `.new_*()` calls in any `forward()` method
- [ ] All model surgery happens in `__init__()` methods
- [ ] All buffers are registered using `register_buffer()`
- [ ] Export completes without TracerWarnings about tensor creation
- [ ] CoreML model loads successfully
- [ ] Inference runs without BNNS crashes
- [ ] Output audio quality matches baseline (if previously working)

## Rollback Criteria

If any of the following occur, revert changes:
1. Export fails with new errors
2. Model size increases by >10%
3. Inference becomes >2x slower
4. Audio quality degrades noticeably

## Success Criteria

1. **Primary**: CoreML inference runs without `EXC_BAD_ACCESS` crashes
2. **Secondary**: No TracerWarnings during export
3. **Bonus**: Can re-enable LSTM paths that were previously bypassed

## Timeline Estimate

- **Setup**: 30 minutes
- **Implementation**: 2-3 hours
- **Testing**: 2 hours
- **Total**: 4-6 hours

## Notes for Engineer

1. The key insight is that PyTorch's `torch.jit.trace` cannot properly handle tensors created dynamically in `forward()`. They must be registered as buffers in `__init__()`.

2. The current workaround (bypassing LSTM with zero F0/N predictions) confirms this diagnosis - they're avoiding the exact code paths that crash.

3. After fixing, you may be able to remove the "bypass LSTM" workaround and use the full model, which should improve audio quality.

4. Use `contiguous()` on expanded buffers to ensure proper memory layout for CoreML.

5. If you encounter new issues, check the MIL graph dump for operations expecting constant shapes but receiving dynamic tensors.


## Implementation Progress

