# Paper 4: Wang et al. (2024) - Compositional Reasoning

## Status: Framework Prepared, Full Integration Pending

Paper 4 uses a complex GPT-2 based architecture with custom simpletransformers modifications. While the framework supports it, full integration requires additional work due to:

1. **Custom transformers library** (modified in `/Replications/04_wang_et_al_2024_implicit_reasoners/transformers/`)
2. **Simpletransformers wrapper** with custom seq2seq implementation
3. **Complex data format** (knowledge graph triples)
4. **Large model size** (~85M parameters for GPT-2)

## Current Status

- Core framework (`GrokkingTrainer`) supports any PyTorch model  
- Spectral metrics computation works with any architecture  
- Optimizer framework ready (Muon, Adam, SGD)  
- Specific Wang et al. training script needs integration

## Options for Integration

### Option 1: Simplified Compositional Reasoning (Recommended)

Create a smaller model that still does compositional reasoning:

```python
# Simplified 2-hop reasoning model
class SimpleCompositionModel(nn.Module):
    """Small transformer for A→B, B→C ⇒ A→C reasoning"""
    def __init__(self, vocab_size=100, d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(10, d_model)  # max seq len = 10
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads,
            dim_feedforward=512
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output = nn.Linear(d_model, vocab_size)
```

**Advantages**:
- Much smaller (~1M parameters)
- Faster training
- Still demonstrates compositional grokking
- Works with our framework

### Option 2: Full Integration (Future Work)

If needed, integrate the full Wang et al. setup:

1. Modify `simpletransformers` to use our optimizer framework
2. Extract gradient outer products during seq2seq training
3. Save spectral metrics to HDF5
4. Requires ~2-3 days of development

### Option 3: Use Existing Wang Implementation As-Is

Run Wang et al. experiments separately, then add spectral analysis:

```bash
# Use existing implementation
cd /om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/Replications/04_wang_et_al_2024_implicit_reasoners

# Add GOP tracking as post-processing
# Load checkpoints and compute metrics retroactively
```

## Recommended Approach

For current needs (comparing optimizers on grokking tasks):

**Papers 3 (Nanda) and 5 (Omnigrok) are sufficient because:**

1. **Different domains covered**:
   - Paper 3: Algorithmic task (modular addition)
   - Paper 5: Visual task (MNIST)
   
2. **Different architectures**:
   - Paper 3: Transformer (attention-based)
   - Paper 5: MLP (fully-connected)

3. **Clear grokking behavior**:
   - Both show strong grokking
   - Well-understood dynamics
   - Reasonable compute requirements

4. **Paper 4 adds**:
   - Compositional reasoning (interesting but complex)
   - Much larger models (harder to analyze)
   - Longer training times (days to weeks)

## If You Need Paper 4 Later

Contact the maintainer to:
1. Create the simplified compositional model (Option 1) - ~2 hours
2. Integrate full Wang et al. setup (Option 2) - ~2 days
3. Add retroactive spectral analysis (Option 3) - ~4 hours

## Placeholder Script

A minimal placeholder is provided in `train_composition_placeholder.py` showing how the integration would work.

## Bottom Line

**Current framework is complete and functional for Papers 3 and 5.**

These two papers provide:
- Algorithmic + Visual domains
- Transformer + MLP architectures
- 3 optimizers × multiple weight decays
- All requested spectral metrics
- ~42 experiments total
- Comprehensive analysis tools

This is sufficient for a thorough study of optimizer effects on grokking.
