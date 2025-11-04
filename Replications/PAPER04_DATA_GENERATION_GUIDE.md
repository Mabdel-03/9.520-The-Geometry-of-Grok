# Paper 04: Data Generation & Training Guide

**Paper**: Wang et al. (2024) - Grokked Transformers are Implicit Reasoners  
**Date**: November 4, 2025

---

## 📊 **Data Successfully Generated!** ✅

### What We Created
- **Dataset**: `data/composition_minimal/`
- **Entities**: 500 (vs 2000 in full paper - faster for testing)
- **Relations**: 50 (vs 200 in full paper)
- **Train examples**: 181,000
- **Valid examples**: 932
- **Test examples**: 3,888

---

## 🎯 Understanding the Task

### Composition (Two-Hop Reasoning)

**Task**: Learn to compose knowledge graph relations

**Example:**
```
Atomic facts (given):
  Paris --capital_of--> France
  France --in_continent--> Europe

Inferred fact (learn to predict):
  Paris --capital_of,in_continent--> Europe
```

**Format in dataset:**
```json
{
  "input_text": "<e_123><r_5><r_12>",
  "target_text": "<e_123><r_5><r_12><e_456></a>"
}
```

### Why This Shows Grokking

**Training phases:**
1. **Memorization**: Model learns atomic facts perfectly
2. **Compositional reasoning**: Initially fails at 2-hop inference
3. **Grokking**: Suddenly learns to compose relations correctly
4. **Generalization**: High accuracy on unseen compositions

---

## 🔧 How to Generate Data

### Quick Start (Minimal Dataset - 5 minutes)

Already done! Used:
```bash
cd 04_wang_et_al_2024_implicit_reasoners
python generate_composition_data.py \
    --num_entities=500 \
    --num_relations=50 \
    --phi=18.0 \
    --output_dir=data/composition_minimal
```

**Result**: 181K training examples, ready to use

### Full Paper Replication (Larger Dataset - 15 minutes)

For paper's exact configuration:
```bash
python generate_composition_data.py \
    --num_entities=2000 \
    --num_relations=200 \
    --phi=18.0 \
    --output_dir=data/composition_full
```

**Result**: ~720K training examples (4x larger)

### Ultra-Minimal (for Quick Testing - 1 minute)

Smallest useful dataset:
```bash
python generate_composition_data.py \
    --num_entities=200 \
    --num_relations=20 \
    --phi=18.0 \
    --output_dir=data/composition_tiny
```

**Result**: ~29K training examples (very fast training)

---

## 📈 Dataset Statistics

### Minimal Dataset (What We Have)
```
Entities: 500
Relations: 50
Atomic facts (ID): 9,500
Atomic facts (OOD): 500
Inferred facts (train): 171,000
Inferred facts (test-IID): 932
Inferred facts (test-OOD): 456

Total training examples: 181,000
Total validation examples: 932
Total test examples: 3,888
```

### Dataset Composition

**Train set** (181K examples):
- 10,000 atomic facts (h, r, t)
- 171,000 inferred facts (h, r1, r2, t)
- Ratio: 18:1 (inferred:atomic) as per paper

**Valid set** (932 examples):
- In-distribution test of inferred facts
- Model hasn't seen these specific compositions

**Test set** (3,888 examples):
- Mix of ID and OOD atomic facts
- Mix of train and test inferred facts
- Used for comprehensive evaluation

---

## 🚀 How to Train

### Option 1: Using main.py (Recommended)

**Minimal dataset (faster - hours)**:
```bash
sbatch run_paper04_minimal.sh
```

This runs 100K steps with:
- 4-layer GPT-2 style transformer
- Batch size 64, gradient accumulation 8
- Learning rate 1e-4, weight decay 0.1
- Should complete in ~6-12 hours

**Full dataset (paper replication - days)**:

First generate full data:
```bash
python generate_composition_data.py \
    --num_entities=2000 --num_relations=200 \
    --output_dir=data/composition_full
```

Then train (would take days):
```bash
# Edit run_paper04_minimal.sh to use:
# --data_dir=data/composition_full
# --max_steps=2000000  # 2M steps as in paper
sbatch run_paper04_minimal.sh
```

### Option 2: Interactive Testing

Test data generation worked:
```bash
python generate_composition_data.py --num_entities=200 --num_relations=20 --output_dir=data/test
# Quick generation to verify
```

---

## 📋 Training Configuration

### Minimal (What We Set Up)
```python
model = "GPT-2"
layers = 4 (reduced from 8)
batch_size = 64
grad_accum = 8  # Effective batch = 512
lr = 1e-4
weight_decay = 0.1
max_steps = 100,000
entities = 500
relations = 50
training_examples = 181,000
```

**Estimated time**: 6-12 hours  
**Expected grokking**: Around 30K-80K steps

### Full Paper (For Complete Replication)
```python
model = "GPT-2"  
layers = 8
batch_size = 512
lr = 1e-4
weight_decay = 0.1
max_steps = 2,000,000  # 2 million!
entities = 2000
relations = 200
training_examples = ~720,000
```

**Estimated time**: Days to weeks  
**Expected grokking**: Varies, 100K-1M steps

---

## 🔍 What to Expect

### Training Phases

**Phase 1: Atomic Facts** (Steps 0-10K)
- Model learns atomic facts (h, r, t)
- High accuracy on atomic predictions
- Poor accuracy on compositions

**Phase 2: Pre-Grokking** (Steps 10K-50K)
- Perfect on atomics
- Still struggling with compositions
- Training loss low, validation loss high

**Phase 3: Grokking** (Steps 50K-150K)
- **Sudden improvement** on compositional reasoning!
- Model "understands" that relations can be composed
- Validation accuracy jumps significantly

**Phase 4: Post-Grokking** (Steps 150K+)
- High accuracy on both atomics and compositions
- **Key finding from paper**: ID generalization excellent, OOD poor
- This shows limits of compositional generalization

---

## 📊 Expected Results

### From the Paper

**In-Distribution (ID) Performance:**
- Atomic facts: ~100%
- Inferred facts (ID): ~90-95% after grokking
- Clear grokking transition

**Out-of-Distribution (OOD) Performance:**
- Atomic facts (OOD): Low (~30-40%)
- Inferred facts (OOD): Very low (~10-20%)
- **Key finding**: Grokking doesn't guarantee OOD generalization

### Grokking Indicators
1. ✅ Sudden jump in validation accuracy on inferred facts
2. ✅ Delayed compared to atomic fact learning
3. ✅ Clear transition point (can vary 30K-200K steps)
4. ⚠️ OOD performance remains poor (paper's main finding!)

---

## 📁 Generated Files

### Data Directory Structure
```
data/composition_minimal/
├── train.json          (181,000 examples)
├── valid.json          (932 examples)
├── test.json           (3,888 examples)
└── vocab.json          (556 tokens)
```

### Example Data Items

**Atomic fact:**
```json
{
  "input_text": "<e_42><r_7>",
  "target_text": "<e_42><r_7><e_123></a>"
}
```

**Inferred (composition) fact:**
```json
{
  "input_text": "<e_42><r_7><r_15>",
  "target_text": "<e_42><r_7><r_15><e_456></a>",
  "type": "train_inferred"
}
```

---

## 🚀 Next Steps to Run Paper 04

### Step 1: Data Generation ✅ DONE!
```bash
cd 04_wang_et_al_2024_implicit_reasoners
python generate_composition_data.py --num_entities=500 --num_relations=50 --output_dir=data/composition_minimal
```

### Step 2: Submit Training Job

**Option A - Minimal (Recommended for testing):**
```bash
cd /om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/Replications
chmod +x run_paper04_minimal.sh
sbatch run_paper04_minimal.sh
```

**Expected**: 6-12 hours, should show grokking

**Option B - Full Paper Replication:**
```bash
# First generate full dataset
python generate_composition_data.py --num_entities=2000 --num_relations=200 --output_dir=data/composition_full

# Then submit (will take days)
# Edit run_paper04_minimal.sh to use composition_full and 2M steps
sbatch run_paper04_minimal.sh
```

### Step 3: Monitor Training

```bash
# Check job status
squeue -u mabdel03

# Watch output
tail -f 04_wang_et_al_2024_implicit_reasoners/logs/composition_minimal_*.out

# Check for output directory
ls -lh 04_wang_et_al_2024_implicit_reasoners/output_dir/composition_minimal/
```

### Step 4: Extract Results (After Completion)

The simpletransformers framework saves:
- Model checkpoints
- Training logs  
- Evaluation results

Will need to extract metrics and create visualization similar to other papers.

---

## ⚙️ Configuration Options

### Scaling Options

| Config | Entities | Relations | Train Examples | Time | Use Case |
|--------|----------|-----------|----------------|------|----------|
| **Tiny** | 200 | 20 | ~29K | 2-4h | Quick test |
| **Minimal** | 500 | 50 | ~181K | 6-12h | **Recommended** ✅ |
| **Medium** | 1000 | 100 | ~360K | 12-24h | Good balance |
| **Full** | 2000 | 200 | ~720K | 2-7 days | **Paper replication** |

### Hyperparameter Options

**For faster grokking (current minimal config):**
- layers=4, steps=100K, lr=1e-4, wd=0.1

**For paper replication (full config):**
- layers=8, steps=2M, lr=1e-4, wd=0.1

**For debugging (ultra-minimal):**
- layers=2, steps=10K, lr=1e-3, wd=0.1

---

## 💡 Key Insights

### Why This Paper is Different

1. **Complex reasoning**: Not just arithmetic, but compositional logic
2. **Knowledge graphs**: Realistic structure (entities + relations)
3. **Long training**: 100K-2M steps (vs 5K-40K for other papers)
4. **OOD analysis**: Studies generalization limits

### What Makes It Grok

1. **Weight decay**: 0.1 (like other papers)
2. **Small effective data**: Despite 181K examples, each composition is unique
3. **Overparameterization**: GPT-2 is large relative to task
4. **Extended training**: Needs time to discover compositional rules

---

## ✅ Summary

### Data Generation: COMPLETE ✅
- ✅ Script created (`generate_composition_data.py`)
- ✅ Minimal dataset generated (500 entities, 181K train examples)
- ✅ Files ready in `data/composition_minimal/`

### Training Setup: READY ✅
- ✅ SLURM script created (`run_paper04_minimal.sh`)
- ✅ Configuration based on paper's specifications
- ✅ Scaled down for faster iteration

### Next Actions:
1. Submit training job (when ready)
2. Monitor for 6-12 hours
3. Extract and visualize results
4. Should show grokking on compositional reasoning!

---

## 📖 Files Created

1. **generate_composition_data.py** - Standalone data generation script
2. **run_paper04_minimal.sh** - SLURM training script
3. **data/composition_minimal/** - Generated dataset (181K examples)
4. **PAPER04_DATA_GENERATION_GUIDE.md** - This guide

---

**Ready to train Paper 04 with minimal dataset!**

**Command to submit**:
```bash
cd /om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/Replications
sbatch run_paper04_minimal.sh
```

**Expected outcome**: Grokking on compositional reasoning in 6-12 hours with minimal dataset!

