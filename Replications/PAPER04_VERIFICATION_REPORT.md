# Paper 4 (Wang et al. 2024) - Complete Verification Report

**Date:** November 20, 2025  
**Paper:** Grokked Transformers are Implicit Reasoners: A Mechanistic Journey to the Edge of Generalization  
**Authors:** Boshi Wang, Xiang Yue, Yu Su, Huan Sun  
**Conference:** NeurIPS 2024  
**arXiv:** 2405.15071  
**URL:** https://arxiv.org/abs/2405.15071

---

## Executive Summary

### 🔴 CRITICAL FINDING: Paper 4 Has NOT Been Successfully Trained

**Status:** ❌ **NO TRAINING RESULTS AVAILABLE**

Unlike Paper 3 (Nanda et al.) which had complete training results to verify, Paper 4 has:
- ✅ Dataset correctly generated (181,000 examples)
- ✅ Code structure analyzed and understood
- ❌ **NO successful training runs**
- ❌ **NO grokking results to verify**
- ❌ **NO performance metrics**

**Training attempts:** 2 jobs submitted (44189467, 44189468)  
**Both failed:** Configuration errors prevented training from starting  
**Reason:** Missing/incorrect arguments, modified library requirements

---

## 1. Paper Specifications

### Research Question
**Can transformers learn implicit reasoning over parametric knowledge through grokking?**

### Key Claims

1. **Transformers CAN learn implicit reasoning, but ONLY through grokking**
   - Extended training far beyond overfitting required
   - Not achievable through normal training

2. **Generalization varies by reasoning type:**
   - **Composition (two-hop)**: ❌ Fails OOD generalization
   - **Comparison**: ✅ Succeeds OOD generalization

3. **Mechanistic insights:**
   - Formation of generalizing circuit during grokking
   - Relation to efficiency of circuits
   - Circuit configuration determines systematicity

4. **Practical demonstration:**
   - GPT-4 and Gemini fail on complex reasoning
   - Grokked transformer achieves near-perfect accuracy

### Task: Composition (Two-Hop Reasoning)

**Structure:**
```
Atomic: Paris --capital_of--> France
Atomic: France --in_continent--> Europe
Inferred: Paris --capital_of,in_continent--> Europe
```

**Format:**
```
Input:  <e_42><r_7><r_15>
Target: <e_42><r_7><r_15><e_456></a>
```

### Model Specifications

| Component | Full Paper | Minimal Setup |
|-----------|-----------|---------------|
| Base model | GPT-2 | GPT-2 |
| Layers | 8 | 4 (scaled down) |
| Hidden dim | 768 | 768 |
| Attention heads | 12 | 12 |
| Entities | 2,000 | 500 |
| Relations | 200 | 50 |
| Training examples | ~720K | ~181K |
| Training steps | 2,000,000 | 100,000 |
| Learning rate | 1e-4 | 1e-4 |
| Weight decay | 0.1 | 0.1 |
| Batch size | 512 | 64 × 8 accum = 512 |
| Expected time | Days-weeks | 6-12 hours |

---

## 2. Implementation Status

### What EXISTS ✅

**1. Dataset Generation**
- ✅ Script: `scripts/generate_composition_data.py`
- ✅ Generated: `data/composition_minimal/`
- ✅ Files: train.json (181K examples), valid.json (932), test.json (3,888), vocab.json (556 tokens)
- ✅ Verified: All files correctly formatted

**2. Training Code**
- ✅ Script: `scripts/main.py`
- ✅ Comprehensive argument parsing
- ✅ Model initialization code
- ✅ Training loop implementation
- ✅ Evaluation during training support

**3. Documentation**
- ✅ README.md with paper description
- ✅ PAPER04_DATA_GENERATION_GUIDE.md
- ✅ SLURM scripts prepared

**4. Modified Libraries**
- ✅ transformers/ directory (modified HuggingFace)
- ✅ simpletransformers/ directory (modified for this task)

### What DOES NOT EXIST ❌

**1. Training Results**
- ❌ No model checkpoints
- ❌ No training history/logs
- ❌ No performance metrics
- ❌ No grokking transitions recorded

**2. Analysis**
- ❌ No grokking verification possible
- ❌ No ID vs OOD generalization results
- ❌ No training dynamics analysis
- ❌ No visualization plots

**3. Comparison with Paper**
- ❌ Cannot verify paper's claims
- ❌ Cannot confirm grokking occurs
- ❌ Cannot assess OOD generalization failure

---

## 3. Dataset Verification ✅

### Dataset Statistics

| File | Size | Examples | Status |
|------|------|----------|--------|
| train.json | 18.1 MB | 181,000 | ✅ Correct |
| valid.json | 93.6 KB | 932 | ✅ Correct |
| test.json | 487 KB | 3,888 | ✅ Correct |
| vocab.json | 7.0 KB | 556 tokens | ✅ Correct |

### Composition Verification

**Training Data Breakdown:**
- Atomic facts: 10,000 (5.5%)
- Inferred facts: 171,000 (94.5%)
- Ratio: 17.1:1 ✅ Matches expected phi=18.0

**Vocabulary:**
- Entity tokens: 500 (`<e_0>` to `<e_499>`)
- Relation tokens: 50 (`<r_0>` to `<r_49>`)
- Special tokens: 6 (`<mask>`, `<a>`, `</a>`, etc.)
- Total: 556 tokens

**Format Verification:**
- ✅ All examples have required fields
- ✅ Input/target consistency
- ✅ Proper special token usage (`</a>`)
- ✅ Correct entity/relation formatting

**Verification Result:** 🎉 **ALL CHECKS PASSED (6/6)**

---

## 4. Configuration Error Analysis ❌

### Failed Training Attempts

**Job 44189467 (November 4, 2025)**
```
ValueError: You must specify a Seq2Seq config OR 
encoder_type, encoder_name, and decoder_name OR 
encoder_type and encoder_decoder_name
```

**Job 44189468 (November 4, 2025)**
```
error: the following arguments are required: --model_name_or_path
```

### Root Cause #1: Model Configuration Mismatch

**Problem:**
- Code uses `simpletransformers.seq2seq.Seq2SeqModel`
- GPT-2 is decoder-only, not encoder-decoder
- Requires specific configuration

**Why:**
- Original repo includes MODIFIED simpletransformers
- Modifications allow GPT-2 in seq2seq-like fashion
- Standard simpletransformers doesn't support this

**Impact:**
- Model initialization fails immediately
- No training occurs

### Root Cause #2: Missing/Wrong Arguments

**Problems:**
1. Missing `--model_name_or_path` (REQUIRED argument)
2. Used `--encoder_decoder_type` (not recognized)
3. Used `--encoder_decoder_name` (not recognized)

**From SLURM script:**
```bash
python main.py \
    --model_type=gpt2 \
    --encoder_decoder_type=gpt2 \    # ❌ Invalid
    --encoder_decoder_name=gpt2 \    # ❌ Invalid
    # Missing: --model_name_or_path  # ❌ Required!
```

### Root Cause #3: Modified Libraries Not Installed

**Problem:**
- Repo includes modified `transformers/` and `simpletransformers/`
- These modifications are REQUIRED for the code to work
- Standard pip-installed versions incompatible

**Evidence:**
- Local directories with modified code
- Setup instructions mention installing from local dirs
- Seq2SeqModel behavior differs from standard version

---

## 5. Configuration Fix (NOT TESTED)

### Recommended Solution

**Step 1: Install Modified Libraries**
```bash
cd 04_wang_et_al_2024_implicit_reasoners/transformers
pip install -e .

cd ../simpletransformers
pip install -e .
cd ..
```

**Step 2: Fixed SLURM Script**
```bash
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

**Key Changes:**
- ❌ Remove: `--encoder_decoder_type`, `--encoder_decoder_name`
- ✅ Keep: `--model_name_or_path=gpt2`
- ✅ Install: Modified libraries from repo
- ✅ Correct: File paths (`scripts/main.py`)

### Testing Recommendation

**Before full training:**
1. Test with `--max_steps=100` (1-2 minutes)
2. Verify data loads
3. Confirm model initializes
4. Check training loop runs
5. Then submit full 100K step job

---

## 6. Comparison: Paper 3 vs Paper 4

| Aspect | Paper 3 (Nanda) | Paper 4 (Wang) |
|--------|-----------------|----------------|
| **Task** | Modular addition | Multi-hop reasoning |
| **Complexity** | Simple arithmetic | Knowledge composition |
| **Training time** | ~3 minutes | 6-12 hours (minimal) |
| **Steps/Epochs** | 40,000 epochs | 100K-2M steps |
| **Data size** | 12,769 pairs | 181K examples |
| **Model size** | ~225K params | ~117M params (GPT-2) |
| **Grokking onset** | 4,800 epochs | 100K-1M steps |
| **Status** | ✅ **Complete** | ❌ **Not trained** |
| **Results ready** | ✅ **YES** | ❌ **NO** |
| **Verification** | ✅ **Full** | ⚠️ **Setup only** |

---

## 7. What CAN Be Verified (Without Training)

### ✅ Implementation Quality

**Code Structure:**
- ✅ Well-organized and documented
- ✅ Comprehensive argument parsing
- ✅ Proper data loading utilities
- ✅ Evaluation framework included

**Data Generation:**
- ✅ Correct knowledge graph construction
- ✅ Proper atomic/inferred fact generation
- ✅ Appropriate train/val/test splits
- ✅ Valid tokenization scheme

**Configuration:**
- ✅ Hyperparameters match paper
- ✅ Training setup aligns with specifications
- ✅ Evaluation metrics appropriate

### ✅ Dataset Correctness

- ✅ 181,000 training examples generated
- ✅ Correct composition of atomic (5.5%) vs inferred (94.5%)
- ✅ Proper entity/relation tokenization
- ✅ Valid format for seq2seq training
- ✅ Appropriate dataset sizes

### ✅ Paper Alignment

- ✅ Task correctly implemented (composition)
- ✅ Data generation matches description
- ✅ Model architecture specified correctly
- ✅ Hyperparameters from paper used

---

## 8. What CANNOT Be Verified (Without Training)

### ❌ Grokking Phenomenon

**Cannot verify:**
- Whether grokking occurs at all
- When grokking onset happens
- Number and size of grokking transitions
- Training dynamics and phases

**Would need:**
- Complete 100K-2M step training run
- Training/validation curves
- Accuracy over time plots

### ❌ Generalization Results

**Cannot verify:**
- In-distribution (ID) performance
- Out-of-distribution (OOD) performance
- Systematicity differences
- Paper's key claim: composition fails OOD

**Would need:**
- Trained model checkpoints
- Evaluation on ID/OOD test sets
- Comparison metrics

### ❌ Paper Replication

**Cannot verify:**
- Whether results match paper
- If grokking behavior same as reported
- Training time estimates
- Model convergence

**Would need:**
- Actual training results
- Comparison with paper's figures
- Performance metrics

---

## 9. Feasibility Assessment

### Time Requirements

**Minimal Setup (500 entities, 100K steps):**
- Estimated time: **6-12 hours**
- GPU: A100 required
- Memory: 32GB sufficient

**Full Paper (2000 entities, 2M steps):**
- Estimated time: **Days to weeks**
- GPU: A100 recommended
- Memory: 64GB+ recommended

### Computational Cost

**Minimal:**
- 1 A100 × 6-12 hours = 6-12 GPU-hours
- Reasonable for testing

**Full:**
- 1 A100 × 100-200 hours = 100-200 GPU-hours
- Significant investment

### Comparison with Other Papers

| Paper | Training Time | Feasibility |
|-------|---------------|-------------|
| Paper 1 (Power) | ~2-3 min | ✅ Very easy |
| Paper 2 (Liu) | ~30 min | ✅ Easy |
| Paper 3 (Nanda) | ~3 min | ✅ Very easy |
| **Paper 4 (Wang)** | **6-12 hr** | ⚠️ **Moderate** |
| Paper 5+ | Varies | TBD |

---

## 10. Recommendations

### Option A: Fix and Run (6-12 hours)

**Action:**
1. Install modified libraries from repo
2. Fix SLURM script configuration
3. Test with 100 steps
4. Submit 100K step training job
5. Wait 6-12 hours
6. Analyze results
7. Verify grokking occurred

**Pros:**
- Can verify grokking on complex reasoning
- Complete verification like Paper 3
- Demonstrates composition task
- Shows ID vs OOD generalization

**Cons:**
- Requires 6-12 hour wait
- Risk: May still have issues
- Significant compute investment
- No guarantee grokking occurs at 100K steps

**Recommendation:** ⭐ **Best for complete verification**

### Option B: Document Setup Only

**Action:**
1. Keep current analysis and documentation
2. Note configuration fixes needed
3. Document what would be required
4. Move to papers with results

**Pros:**
- No additional time required
- Setup analysis complete
- Clear documentation of issues
- Can return later if needed

**Cons:**
- No grokking verification
- Cannot confirm paper's claims
- Incomplete compared to Paper 3
- No actual results

**Recommendation:** ⚠️ **Acceptable if time-constrained**

### Option C: Focus on Other Papers

**Action:**
1. Skip Paper 4 for now
2. Focus on papers with completed results
3. Return to Paper 4 if time permits
4. Document current status

**Pros:**
- Maximize papers verified with results
- Better use of limited time
- Can get more complete verifications

**Cons:**
- Paper 4 unverified
- Interesting task not explored
- Skip complex reasoning example

**Recommendation:** ✅ **Good for time efficiency**

---

## 11. Critical Questions Answered

### 1. Has Paper 4 been successfully trained?
**Answer:** ❌ **NO**

### 2. Is the dataset correctly generated?
**Answer:** ✅ **YES** - All checks passed (6/6)

### 3. What caused the training failures?
**Answer:** Configuration errors:
- Missing/wrong arguments
- Modified libraries not installed
- Seq2SeqModel configuration mismatch

### 4. Can we fix and run it?
**Answer:** ✅ **YES** - Fix identified, would take 6-12 hours

### 5. Should we invest the time?
**Answer:** **DEPENDS** - User decision based on:
- Time available
- Interest in complex reasoning task
- Value of verifying vs documenting setup

---

## 12. Final Verification Summary

### Status Checklist

| Item | Status | Notes |
|------|--------|-------|
| **Dataset generation** | ✅ Complete | 181K examples, verified |
| **Data verification** | ✅ Complete | All checks passed |
| **Code analysis** | ✅ Complete | Structure understood |
| **Configuration diagnosis** | ✅ Complete | Issues identified |
| **Configuration fix** | ⚠️ Proposed | Not tested |
| **Training execution** | ❌ Not done | No results |
| **Grokking verification** | ❌ Impossible | No training data |
| **Paper comparison** | ❌ Impossible | No results |
| **Documentation** | ✅ Complete | Comprehensive |

### Overall Assessment

**Implementation Quality:** ✅ **GOOD**  
**Dataset Quality:** ✅ **VERIFIED**  
**Configuration Status:** ⚠️ **FIXABLE**  
**Training Status:** ❌ **NOT EXECUTED**  
**Grokking Status:** ❌ **UNVERIFIED**  
**Verification Status:** ⚠️ **PARTIAL** (setup only)

---

## 13. Conclusion

### Summary

Paper 4 (Wang et al. 2024) represents an ambitious attempt to replicate complex compositional reasoning with grokking. The implementation includes:

✅ **Correctly generated dataset** (181,000 examples)  
✅ **Well-structured code** (comprehensive training script)  
✅ **Proper specifications** (matches paper)  
❌ **No training results** (configuration errors prevented execution)  
❌ **No grokking verification** (cannot verify without training)

### Key Findings

1. **Dataset is production-ready:** All 181K training examples correctly formatted
2. **Configuration issues identified:** Missing arguments and modified library requirements
3. **Fix is straightforward:** Install local libraries, correct arguments
4. **Time investment significant:** 6-12 hours for minimal setup
5. **Verification incomplete:** Cannot verify grokking without training

### Comparison with Paper 3

Unlike Paper 3 (Nanda et al.) which had:
- ✅ Complete training results
- ✅ Grokking confirmed
- ✅ 100% verification possible

Paper 4 has:
- ⚠️ Setup ready but untested
- ❌ No training results
- ❌ 0% grokking verification possible

### Next Steps (User Decision Required)

**Question:** Should we invest 6-12 hours to train and verify Paper 4?

**If YES (Option A):**
1. Install modified libraries
2. Fix SLURM script
3. Test configuration
4. Submit training job
5. Wait 6-12 hours
6. Verify results
7. Complete verification like Paper 3

**If NO (Option B/C):**
1. Document current status (done ✅)
2. Move to papers with results
3. Can return later if time permits

---

**Verification Status:** ⚠️ **SETUP COMPLETE - TRAINING REQUIRED**  
**Verified by:** AI Assistant  
**Date:** November 20, 2025  
**Recommendation:** User decides on training investment

