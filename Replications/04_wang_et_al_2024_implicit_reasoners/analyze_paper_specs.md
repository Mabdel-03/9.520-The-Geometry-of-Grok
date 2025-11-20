# Paper 4: Grokked Transformers are Implicit Reasoners - Specifications

**Authors:** Boshi Wang, Xiang Yue, Yu Su, Huan Sun  
**Conference:** NeurIPS 2024  
**arXiv:** 2405.15071  
**URL:** https://arxiv.org/abs/2405.15071

## Key Claims from Abstract

### Main Research Question
Can transformers learn to implicitly reason over parametric knowledge?

### Key Findings

1. **Transformers CAN learn implicit reasoning, but ONLY through grokking**
   - Extended training far beyond overfitting required
   - Not achievable through normal training

2. **Generalization varies by reasoning type:**
   - **Composition (two-hop)**: Fails to systematically generalize OOD
   - **Comparison**: Succeeds at systematic OOD generalization

3. **Mechanistic insights discovered:**
   - Formation of generalizing circuit during grokking
   - Relation to efficiency of generalizing vs memorizing circuits
   - Connection between systematicity and circuit configuration

4. **Practical demonstration:**
   - GPT-4-Turbo and Gemini-1.5-Pro fail badly on complex reasoning
   - Fully grokked transformer achieves near-perfect accuracy
   - Shows power of parametric memory for complex reasoning

## Task Specifications

### Composition Task (Two-Hop Reasoning)

**Structure:**
- Knowledge graph with entities and relations
- Atomic facts: (head, relation, tail)
- Composed facts: (head, relation1, relation2, tail)

**Example:**
```
Atomic: Paris -capital_of-> France
Atomic: France -in_continent-> Europe
Composed: Paris -capital_of,in_continent-> Europe
```

**Datasets:**
- Entities: 2000 (full), 500 (minimal)
- Relations: 200 (full), 50 (minimal)
- Training examples: ~720K (full), ~181K (minimal)

### Comparison Task

**Structure:**
- Compare attributes of entities
- Learn comparison relations

## Model Specifications

**Architecture:**
- Base: GPT-2 style transformer
- Layers: 8 (full paper)
- Hidden dim: 768 (GPT-2 default)
- Attention heads: 12 (GPT-2 default)
- Modified for task-specific tokenization

**Training Configuration:**
- Optimizer: Adam
- Learning rate: 1e-4
- Weight decay: 0.1 (critical for grokking)
- Batch size: 512
- Max steps: 2,000,000 (2M for full)
- Sequence length: 10 tokens

## Expected Training Dynamics

### Phase 1: Memorization (Steps 0-10K)
- Learn atomic facts
- Poor compositional accuracy

### Phase 2: Extended Training (Steps 10K-100K)
- Perfect on atomics
- Still poor on compositions
- High training/validation gap

### Phase 3: Grokking (Steps 100K-1M)
- **Sudden improvement** on compositions
- Discovery of compositional circuit
- Validation accuracy jumps

### Phase 4: Post-Grokking (Steps 1M+)
- High accuracy on ID compositions
- **Poor OOD generalization** (key finding)

## Expected Results

### In-Distribution (ID) Performance
- Atomic facts: ~100%
- Composed facts (ID): ~90-95% after grokking
- Clear delayed generalization

### Out-of-Distribution (OOD) Performance
- **Critical Finding**: Grokking ≠ OOD generalization
- Composition OOD: ~10-30% (poor)
- Comparison OOD: ~80-90% (good)
- Shows task-dependent systematicity

## Key Differences from Paper 3 (Nanda et al.)

| Aspect | Paper 3 (Nanda) | Paper 4 (Wang) |
|--------|-----------------|----------------|
| Task | Modular addition | Multi-hop reasoning |
| Complexity | Simple arithmetic | Knowledge composition |
| Training | 40K epochs | 2M steps |
| Time | ~3 minutes | Days to weeks |
| Grokking onset | 4,800 epochs | 100K-1M steps |
| OOD generalization | N/A | **Fails** (key finding) |
| Model size | ~225K params | ~117M params (GPT-2) |

## Critical Hyperparameters

**For Grokking to Occur:**
1. Weight decay = 0.1 (CRITICAL)
2. Extended training (100K+ steps minimum)
3. Overparameterized model (GPT-2 size)
4. Full batch or large batch size (512)

**For Faster Iteration (Minimal Setup):**
- Entities: 500 (vs 2000)
- Relations: 50 (vs 200)
- Layers: 4 (vs 8)
- Steps: 100K (vs 2M)
- Expected time: 6-12 hours (vs days)

## Mechanistic Analysis

### Logit Lens
- Track layer-by-layer predictions
- Identify where reasoning occurs

### Causal Tracing
- Measure importance of different layers
- Map formation of generalizing circuit

### Circuit Discovery
- Grokking corresponds to circuit formation
- Different circuits for composition vs comparison
- Explains OOD generalization differences

## Significance

**Why This Paper Matters:**
1. Shows grokking on complex reasoning (not just arithmetic)
2. Reveals limits: grokking ≠ systematic generalization
3. Provides mechanistic understanding of reasoning
4. Demonstrates parametric > non-parametric for complex tasks

**Implications:**
- Training time critical for reasoning tasks
- Need to design for OOD generalization explicitly
- Circuit architecture determines systematicity
- Weight decay essential for discovering generalizable solutions

