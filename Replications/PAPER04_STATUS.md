# Paper 04 Status: Wang et al. (2024) - Implicit Reasoners

**Status**: ⚠️ **COMPLEX - SKIP FOR NOW**  
**Priority**: Low (return if time permits)

---

## Issues Found

### 1. Missing Directory Structure
- ❌ No `src/` directory (mentioned in README)
- ❌ No dedicated `data/` directory with pre-generated knowledge graphs
- Has `main.py` in root but requires many arguments

### 2. Complex Dependencies
- Custom `simpletransformers` implementation (included in repo)
- Modified `transformers` library (included in repo)
- Requires knowledge graph data generation

### 3. Scale and Complexity
- **Training time**: Days to weeks
- **Max steps**: 2,000,000 (2 million!)
- **Task**: Complex knowledge graph reasoning (not simple modular arithmetic)
- **Model**: GPT-2 style transformer (8 layers, 768 dim)

### 4. Required Arguments
`main.py` requires:
- `--data_dir` (with train.json, valid.json, test.json)
- `--model_name_or_path`
- Many architectural and training flags

---

## What the Paper Does

**Task**: Knowledge graph reasoning
- **Composition**: Two-hop reasoning (e.g., if A→B and B→C, then A→C)
- **Comparison**: Entity comparison based on attributes

**Key Finding**: 
- Grokking occurs on both tasks
- But OOD generalization differs:
  - Composition: Grokking but poor OOD
  - Comparison: Grokking with good OOD

**Why it matters**: Shows grokking on **complex reasoning** tasks, not just arithmetic

---

## Why We're Skipping (For Now)

1. **Data generation needed**: No pre-made datasets
2. **Very long training**: 2M steps = days/weeks
3. **Complex setup**: Custom transformers implementation
4. **Missing components**: No src/ directory as documented
5. **High debugging effort**: Would take hours just to set up

---

## If We Return to This

### Option 1: Use Notebooks
The repository has Jupyter notebooks (composition.ipynb, comparison.ipynb) that might be simpler:
```bash
cd 04_wang_et_al_2024_implicit_reasoners
jupyter notebook composition.ipynb
```

### Option 2: Generate Data First
Need to create knowledge graph datasets before training

### Option 3: Simplified Version
Create a minimal composition task with smaller graphs (100 entities vs 2000)

---

## Recommendation

**Skip Paper 04 for now** because:
1. We already have 3 confirmed grokking results
2. Papers 07, 08 completed but didn't learn - quick fixes likely
3. Papers 06, 09, 10 have simpler setups
4. Paper 01 needs PyTorch Lightning migration (medium effort)

**Return to Paper 04** only if:
- We get 7-8 other papers working
- We want to show grokking on complex reasoning tasks
- We have several days for training

---

**Decision**: Move to Papers 06-10 which are more tractable.

**Next Focus**: Papers 07 & 08 (completed but models didn't learn - likely quick hyperparameter fixes)

