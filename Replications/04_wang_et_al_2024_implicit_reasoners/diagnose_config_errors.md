# Paper 4: Configuration Error Diagnosis

## Error Log Analysis

### Job 44189467 - Error Log
```
Traceback (most recent call last):
  File "main.py", line 197, in <module>
    main()
  File "main.py", line 161, in main
    model = Seq2SeqModel(
  File ".../simpletransformers/seq2seq/seq2seq_model.py", line 169, in __init__
    raise ValueError(
ValueError: You must specify a Seq2Seq config OR encoder_type, encoder_name, and decoder_name OR encoder_type and encoder_decoder_name
```

### Job 44189468 - Error Log
```
usage: main.py [-h] --data_dir DATA_DIR [--model_type MODEL_TYPE]
               --model_name_or_path MODEL_NAME_OR_PATH ...
main.py: error: the following arguments are required: --model_name_or_path
```

## Root Cause Analysis

### Error 1: Seq2SeqModel Configuration Issue

**What Happened:**
- Script called `Seq2SeqModel(model_type='gpt2', model_name=args.model_name_or_path, ...)`
- Simpletransformers Seq2SeqModel expects specific encoder/decoder configuration
- GPT-2 is a decoder-only model, not a true encoder-decoder seq2seq model

**Why It Failed:**
The Seq2SeqModel __init__ (line 169) checks:
```python
if not (config OR (encoder_type AND encoder_name AND decoder_name) OR 
        (encoder_type AND encoder_decoder_name)):
    raise ValueError(...)
```

Our call provided:
- `model_type='gpt2'` 
- `model_name='gpt2'` (if --model_name_or_path was present)
- But NO encoder_type, encoder_name, decoder_name

**The Issue:**
- The original paper's repo includes MODIFIED versions of simpletransformers
- These modifications likely allow GPT-2 to be used in a seq2seq-like fashion
- Our standard simpletransformers installation doesn't have these modifications

### Error 2: Missing Required Argument

**What Happened:**
- Second job attempt removed some arguments to simplify
- But removed `--model_name_or_path` which is REQUIRED in main.py (line 16)
- ArgumentParser raised error before model initialization

**Current SLURM Script Issues:**

```bash
# Current broken script (run_paper04_minimal.sh)
python main.py \
    --model_type=gpt2 \
    --encoder_decoder_type=gpt2 \      # ❌ Not a valid argument
    --encoder_decoder_name=gpt2 \      # ❌ Not recognized by argparse
    --model_name_or_path=gpt2          # ✅ This line was missing in job 2
    ...
```

## Solution Options

### Option 1: Use Modified Simpletransformers (Recommended)
**Action:** Use the simpletransformers/ directory in the repo
**Pros:** 
- Intended way to run the code
- Modified to support this use case
- Most likely to work

**Cons:**
- Need to install local version
- May have compatibility issues

**Implementation:**
```bash
cd 04_wang_et_al_2024_implicit_reasoners/simpletransformers
pip install -e .
```

### Option 2: Fix Configuration for Standard Simpletransformers
**Action:** Provide correct encoder/decoder arguments
**Pros:**
- Uses standard library
- More portable

**Cons:**
- May not work as intended
- GPT-2 isn't really a seq2seq model

**Implementation:**
```bash
python main.py \
    --data_dir=data/composition_minimal \
    --model_type=gpt2 \
    --encoder_decoder_type=gpt2 \
    --encoder_decoder_name=gpt2 \
    --model_name_or_path=gpt2 \
    ...
```

### Option 3: Modify main.py to Use LanguageModelingModel
**Action:** Change to use LanguageModelingModel instead
**Pros:**
- More appropriate for decoder-only GPT-2
- Standard simpletransformers

**Cons:**
- Requires code modification
- Changes paper's original approach
- May not support seq2seq-style data format

## Recommended Fix

**Use the repo's modified simpletransformers:**

```bash
#!/bin/bash
# Updated SLURM script

cd /path/to/04_wang_et_al_2024_implicit_reasoners

# Install modified simpletransformers from repo
cd simpletransformers
pip install -e . --quiet
cd ..

# Run with correct arguments
python scripts/main.py \
    --data_dir=data/composition_minimal \
    --model_type=gpt2 \
    --model_name_or_path=gpt2 \
    --init_weights \
    --n_layer=4 \
    --add_tokens \
    --no_dropout \
    --do_train \
    --do_eval \
    --evaluate_during_training \
    --overwrite_output_dir \
    --output_dir=output_dir/composition_minimal \
    --train_batch_size=64 \
    --eval_batch_size=64 \
    --gradient_accumulation_steps=8 \
    --learning_rate=1e-4 \
    --weight_decay=0.1 \
    --max_steps=100000 \
    --save_step=10000 \
    --warmup_steps=1000 \
    --scheduler=linear_schedule_with_warmup \
    --max_seq_length=64 \
    --max_length=64 \
    --manual_seed=42
```

## Key Changes from Broken Script

1. **Install modified simpletransformers:**
   ```bash
   cd simpletransformers && pip install -e . && cd ..
   ```

2. **Remove invalid arguments:**
   - ❌ Remove `--encoder_decoder_type=gpt2`
   - ❌ Remove `--encoder_decoder_name=gpt2`

3. **Keep required arguments:**
   - ✅ Keep `--model_name_or_path=gpt2`
   - ✅ Keep `--data_dir=data/composition_minimal`

4. **Correct file paths:**
   - Change `main.py` to `scripts/main.py`
   - Ensure data_dir path is correct

## Testing Strategy

**Before submitting 6-12 hour job:**

1. **Quick syntax check:**
   ```bash
   python scripts/main.py --help
   ```

2. **Verify data loading:**
   ```bash
   python scripts/main.py \
       --data_dir=data/composition_minimal \
       --model_name_or_path=gpt2 \
       --max_steps=10 \
       --do_train
   ```

3. **Test full config (dry run):**
   - Run for 100 steps (~1-2 minutes)
   - Verify training loop works
   - Check checkpoint saving
   - Then submit full 100K step job

## Expected Behavior After Fix

**Successful Training Start:**
```
Loading data from data/composition_minimal/train.json...
Loaded 181,000 training examples
Adding 556 new tokens to vocabulary
Initializing model with 4 layers...
Starting training for 100,000 steps...
Step 100: loss=4.235, train_acc=0.023
...
```

**Signs Training Works:**
- No ValueError on model initialization
- Data loads successfully
- Training loss decreases
- Periodic evaluation runs
- Checkpoints saved

## Diagnosis Summary

✅ **Dataset:** Correctly generated and validated  
❌ **Configuration:** Broken due to missing/wrong arguments  
❌ **Simpletransformers:** Need modified version from repo  
✅ **Fix Identified:** Install local simpletransformers, use correct args  
⏳ **Status:** Ready to fix and test

