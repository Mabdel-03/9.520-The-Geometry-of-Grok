# Paper 4: Code Structure Analysis

## Main Training Script (main.py)

### Key Observations

**1. Model Initialization (Lines 1-197)**
- Uses `simpletransformers.seq2seq.Seq2SeqModel`
- **CRITICAL**: Requires specific configuration for encoder-decoder models
- GPT-2 is NOT a standard seq2seq model (it's decoder-only)
- This is likely the source of configuration errors

**2. Required Arguments**
-`--model_name_or_path` (REQUIRED) - Missing in our SLURM script!
- `--data_dir` (REQUIRED) - Present ✅
- `--model_type` - Default 'gpt2' ✅

**3. Model Configuration Options**
- `--init_weights`: Fresh weight initialization
- `--n_layer`: Number of layers (we use 4)
- `--n_head`: Number of attention heads
- `--add_tokens`: Add custom tokens from vocab.json
- `--no_dropout`: Disable dropout
- `--weight_decay`: Default 0.01, paper uses 0.1

**4. Training Hyperparameters**
- `--train_batch_size`: Default 16, paper uses 512
- `--eval_batch_size`: Default 16
- `--gradient_accumulation_steps`: For effective larger batch
- `--learning_rate`: Default 4e-5, paper uses 1e-4
- `--max_steps`: For long training (we want 100K)
- `--scheduler`: Type of LR schedule

**5. Evaluation Options**
- `--evaluate_during_training`: Track progress
- `--save_step`: Checkpoint frequency
- `--save_best_model`: Keep best model

### Issue Diagnosis

**Error 1**: `ValueError: You must specify a Seq2Seq config OR encoder_type, encoder_name, and decoder_name`

**Root Cause**:
- Simpletransformers Seq2SeqModel expects encoder-decoder configuration
- GPT-2 is decoder-only, not true seq2seq
- Need proper configuration or different model class

**Error 2**: Missing `--model_name_or_path`

**Root Cause**:
- Argument is REQUIRED in main.py (line 16)
- Our SLURM script used `--encoder_decoder_name` instead
- Wrong argument name

### Data Generation Script (generate_composition_data.py)

**Functionality**:
1. Creates synthetic knowledge graph
2. Generates atomic facts: (h, r, t)
3. Generates composed facts: (h, r1, r2, t)
4. Splits into train/validation/test sets
5. Creates vocab with entity/relation tokens

**Key Parameters**:
- `num_entities`: 500 (our minimal) vs 2000 (paper)
- `num_relations`: 50 (our minimal) vs 200 (paper)
- `out_degree`: 20 (edges per entity)
- `phi`: 18.0 (ratio of inferred to atomic facts)

**Output Files**:
- `train.json`: Training examples
- `valid.json`: Validation examples
- `test.json`: Test examples
- `vocab.json`: Custom tokens

## Configuration Issues Summary

### Issue #1: Model Type Mismatch
**Problem**: Using Seq2SeqModel for decoder-only GPT-2
**Impact**: Model fails to initialize
**Fix Options**:
1. Use LanguageModelingModel instead of Seq2SeqModel
2. Configure Seq2SeqModel correctly for decoder-only
3. Use transformers directly (bypass simpletransformers)

### Issue #2: Missing Required Argument
**Problem**: `--model_name_or_path` not provided
**Impact**: ArgumentParser raises required argument error
**Fix**: Add `--model_name_or_path=gpt2` to SLURM script

### Issue #3: Wrong Argument Names
**Problem**: Used `--encoder_decoder_type` and `--encoder_decoder_name`
**Impact**: Arguments not recognized
**Fix**: Use correct argument names from main.py

## Required Fixes for SLURM Script

**Current (Broken)**:
```bash
python main.py \
    --model_type=gpt2 \
    --encoder_decoder_type=gpt2 \
    --encoder_decoder_name=gpt2 \
    ...
```

**Fixed (Should Work)**:
```bash
python main.py \
    --model_type=gpt2 \
    --model_name_or_path=gpt2 \
    ...
```

**Additional Considerations**:
- May need to verify simpletransformers supports this usage
- Original repo might have modified simpletransformers
- Check if custom modifications exist in repo's simpletransformers/

## Code Quality Assessment

### Strengths
✅ Data generation is well-structured
✅ Comprehensive argument parsing
✅ Supports various model configurations
✅ Includes evaluation during training
✅ Checkpoint saving implemented

### Concerns
⚠️ Simpletransformers dependency complexity
⚠️ Seq2SeqModel for decoder-only task unclear
⚠️ Modified libraries (transformers/, simpletransformers/) in repo
⚠️ No standalone training example in README
⚠️ Long training time requirement (100K-2M steps)

## Next Steps

1. **Verify Dataset**: Check generated data files are correct
2. **Fix Configuration**: Create corrected SLURM script
3. **Test Locally**: Try short training run to verify config
4. **Decide**: Whether to invest 6-12 hours in training

