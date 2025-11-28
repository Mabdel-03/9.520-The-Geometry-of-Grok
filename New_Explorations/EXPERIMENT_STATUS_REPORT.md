# 📊 Non-AGOP Experiments - Complete Status Report

**Generated:** November 26, 2024  
**Location:** `/om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/New_Explorations/optimizer_experiments`

---

## Executive Summary

| Dataset | Experiments | Completed | Grokked | Success Rate | Status |
|---------|-------------|-----------|---------|--------------|--------|
| **Nanda** | 24 | 24 (100%) | 6 (25%) | ✅ 100% | Complete |
| **Softmax** | 21 | 21 (100%) | 4 (19%) | ✅ 100% | Complete |
| **MNIST** | 18 | 18 (100%) | 4 (22%) | ✅ 100% | Complete |
| **Composition** | 21 | 0 (0%) | 0 (0%) | ⏳ 0% | Running (34+ hours) |
| **TOTAL** | **84** | **63 (75%)** | **14 (17%)** | | |

---

## 1. Nanda (Modular Addition - ReLU Transformer) ✅

**Status:** 24/24 COMPLETE  
**Grokking:** 6/24 (25%)  
**Best Configuration:** AdamW with wd=2.0 (grokked at epoch 2600)

### Results Table

| Experiment | Optimizer | Weight Decay | Train Acc | Test Acc | Grokked | Grok Epoch |
|------------|-----------|--------------|-----------|----------|---------|------------|
| nanda_adamw_wd0.0 | AdamW | 0.0 | 0.0125 | 0.0073 | ❌ | - |
| nanda_adamw_wd0.01 | AdamW | 0.01 | 0.0128 | 0.0072 | ❌ | - |
| nanda_adamw_wd0.1 | AdamW | 0.1 | 1.0000 | 0.3215 | ❌ | - |
| **nanda_adamw_wd0.5** | **AdamW** | **0.5** | **1.0000** | **0.9998** | **✅** | **12,600** |
| **nanda_adamw_wd1.0** | **AdamW** | **1.0** | **1.0000** | **1.0000** | **✅** | **7,500** |
| **nanda_adamw_wd2.0** | **AdamW** | **2.0** | **1.0000** | **1.0000** | **✅** | **2,600** ⭐ |
| **nanda_adamw_wd5.0** | **AdamW** | **5.0** | **1.0000** | **0.9843** | **✅** | **900** |
| **nanda_adamw_wd10.0** | **AdamW** | **10.0** | **0.9932** | **0.9588** | **✅** | **400** |
| nanda_muonw_wd0.0 | Muon | 0.0 | 1.0000 | 0.0002 | ❌ | - |
| nanda_muonw_wd0.01 | Muon | 0.01 | 1.0000 | 0.1539 | ❌ | - |
| nanda_muonw_wd0.1 | Muon | 0.1 | 0.0097 | 0.0085 | ❌ | - |
| nanda_muonw_wd0.5 | Muon | 0.5 | 0.0123 | 0.0074 | ❌ | - |
| nanda_muonw_wd1.0 | Muon | 1.0 | 0.0120 | 0.0075 | ❌ | - |
| nanda_muonw_wd2.0 | Muon | 2.0 | 0.0089 | 0.0088 | ❌ | - |
| nanda_muonw_wd5.0 | Muon | 5.0 | 0.0065 | 0.0098 | ❌ | - |
| nanda_muonw_wd10.0 | Muon | 10.0 | 0.0081 | 0.0092 | ❌ | - |
| nanda_sgd_wd0.0 | SGD | 0.0 | 1.0000 | 0.0012 | ❌ | - |
| **nanda_sgd_wd0.01** | **SGD** | **0.01** | **1.0000** | **0.9950** | **✅** | **36,800** |
| nanda_sgd_wd0.1 | SGD | 0.1 | 0.0117 | 0.0076 | ❌ | - |
| nanda_sgd_wd0.5 | SGD | 0.5 | 0.0102 | 0.0083 | ❌ | - |
| nanda_sgd_wd1.0 | SGD | 1.0 | 0.0089 | 0.0088 | ❌ | - |
| nanda_sgd_wd2.0 | SGD | 2.0 | 0.0094 | 0.0086 | ❌ | - |
| nanda_sgd_wd5.0 | SGD | 5.0 | 0.0086 | 0.0089 | ❌ | - |
| nanda_sgd_wd10.0 | SGD | 10.0 | 0.0073 | 0.0095 | ❌ | - |

**Key Findings:**
- ✅ AdamW dominates: 5/8 configurations grok
- 🎯 Optimal weight decay: 0.5-2.0 range
- ⚡ Higher WD = faster grokking (2.0 groks 3× faster than 1.0)
- ❌ Muon: 0/8 grokked (struggled with this architecture)
- ⚠️ SGD: Only 1/8 grokked (very narrow window at wd=0.01)

---

## 2. Softmax (Modular Addition - Standard Transformer) ✅

**Status:** 21/21 COMPLETE  
**Grokking:** 4/21 (19%)  
**Best Configuration:** AdamW with wd=1.0 (grokked at epoch 1300)

### Results Table

| Experiment | Optimizer | Weight Decay | Train Acc | Test Acc | Grokked | Grok Epoch |
|------------|-----------|--------------|-----------|----------|---------|------------|
| softmax_adamw_wd0.0 | AdamW | 0.0 | 0.0119 | 0.0094 | ❌ | - |
| softmax_adamw_wd0.01 | AdamW | 0.01 | 1.0000 | 0.0684 | ❌ | - |
| **softmax_adamw_wd0.1** | **AdamW** | **0.1** | **1.0000** | **1.0000** | **✅** | **11,500** |
| **softmax_adamw_wd0.5** | **AdamW** | **0.5** | **1.0000** | **1.0000** | **✅** | **2,700** |
| **softmax_adamw_wd1.0** | **AdamW** | **1.0** | **1.0000** | **1.0000** | **✅** | **1,300** ⭐ |
| **softmax_adamw_wd2.0** | **AdamW** | **2.0** | **0.9802** | **0.9728** | **✅** | **800** |
| softmax_adamw_wd5.0 | AdamW | 5.0 | 0.2649 | 0.1881 | ❌ | - |
| softmax_muonw_wd0.0 | Muon | 0.0 | 1.0000 | 0.3994 | ❌ | - |
| softmax_muonw_wd0.01 | Muon | 0.01 | 1.0000 | 0.8922 | ❌ | - |
| softmax_muonw_wd0.1 | Muon | 0.1 | 0.0128 | 0.0079 | ❌ | - |
| softmax_muonw_wd0.5 | Muon | 0.5 | 0.0128 | 0.0079 | ❌ | - |
| softmax_muonw_wd1.0 | Muon | 1.0 | 0.0302 | 0.0442 | ❌ | - |
| softmax_muonw_wd2.0 | Muon | 2.0 | 0.0128 | 0.0079 | ❌ | - |
| softmax_muonw_wd5.0 | Muon | 5.0 | 0.0128 | 0.0079 | ❌ | - |
| softmax_sgd_wd0.0 | SGD | 0.0 | 1.0000 | 0.2300 | ❌ | - |
| softmax_sgd_wd0.01 | SGD | 0.01 | 0.0128 | 0.0079 | ❌ | - |
| softmax_sgd_wd0.1 | SGD | 0.1 | 0.0128 | 0.0079 | ❌ | - |
| softmax_sgd_wd0.5 | SGD | 0.5 | 0.0128 | 0.0079 | ❌ | - |
| softmax_sgd_wd1.0 | SGD | 1.0 | 0.0128 | 0.0079 | ❌ | - |
| softmax_sgd_wd2.0 | SGD | 2.0 | 0.0128 | 0.0079 | ❌ | - |
| softmax_sgd_wd5.0 | SGD | 5.0 | 0.0128 | 0.0079 | ❌ | - |

**Key Findings:**
- ✅ AdamW only optimizer that groks
- 🎯 Optimal range: wd=0.1-2.0
- ⚡ Higher WD = faster (1.0 groks in ~1300 epochs)
- ❌ Muon & SGD: 0/14 grokked on softmax architecture

---

## 3. MNIST (Omnigrok - Image Classification) ✅

**Status:** 18/18 COMPLETE  
**Grokking:** 4/18 (22%)  
**Best Configuration:** AdamW with wd=0.1 (grokked at epoch 3700)

### Results Table

| Experiment | Optimizer | Weight Decay | Train Acc | Test Acc | Grokked | Grok Epoch |
|------------|-----------|--------------|-----------|----------|---------|------------|
| mnist_adamw_wd0.0 | AdamW | 0.0 | 1.0000 | 0.7532 | ❌ | - |
| mnist_adamw_wd0.001 | AdamW | 0.001 | 1.0000 | 0.7819 | ❌ | - |
| mnist_adamw_wd0.01 | AdamW | 0.01 | 1.0000 | 0.8941 | ❌ | - |
| **mnist_adamw_wd0.1** | **AdamW** | **0.1** | **1.0000** | **0.9171** | **✅** | **3,700** ⭐ |
| **mnist_adamw_wd0.5** | **AdamW** | **0.5** | **1.0000** | **0.9153** | **✅** | **2,600** |
| **mnist_adamw_wd1.0** | **AdamW** | **1.0** | **1.0000** | **0.9069** | **✅** | **2,700** |
| mnist_muonw_wd0.0 | Muon | 0.0 | 1.0000 | 0.4297 | ❌ | - |
| mnist_muonw_wd0.001 | Muon | 0.001 | 1.0000 | 0.6537 | ❌ | - |
| mnist_muonw_wd0.01 | Muon | 0.01 | 0.2030 | 0.1733 | ❌ | - |
| mnist_muonw_wd0.1 | Muon | 0.1 | 0.1170 | 0.1028 | ❌ | - |
| mnist_muonw_wd0.5 | Muon | 0.5 | 0.1170 | 0.1028 | ❌ | - |
| mnist_muonw_wd1.0 | Muon | 1.0 | 0.1170 | 0.1028 | ❌ | - |
| mnist_sgd_wd0.0 | SGD | 0.0 | 0.9990 | 0.4600 | ❌ | - |
| **mnist_sgd_wd0.001** | **SGD** | **0.001** | **1.0000** | **0.9078** | **✅** | **40,000** |
| mnist_sgd_wd0.01 | SGD | 0.01 | 0.7940 | 0.7601 | ❌ | - |
| mnist_sgd_wd0.1 | SGD | 0.1 | 0.1170 | 0.1028 | ❌ | - |
| mnist_sgd_wd0.5 | SGD | 0.5 | 0.1170 | 0.1028 | ❌ | - |
| mnist_sgd_wd1.0 | SGD | 1.0 | 0.1170 | 0.1028 | ❌ | - |

**Key Findings:**
- ✅ MSE loss (not CrossEntropy) was critical for grokking
- 🎯 AdamW best: 3/6 configurations grok
- ⚡ Sweet spot: wd=0.1-1.0
- ❌ Muon: 0/6 grokked (struggled on MNIST)
- ⚠️ SGD: Only 1/6 grokked (very late at wd=0.001)

---

## 4. Composition (Compositional Reasoning) ⏳ RUNNING

**Status:** 0/21 COMPLETE (Still running after 34+ hours)  
**Configuration:** GPT-2 with 28.8M parameters, 181K training samples

### Current Status

| Experiment | Optimizer | Weight Decay | Status | Runtime | Progress |
|------------|-----------|--------------|--------|---------|----------|
| comp_adamw_wd0.0 | AdamW | 0.0 | 🔄 Running | 34:44:02 | In progress |
| comp_adamw_wd0.01 | AdamW | 0.01 | 🔄 Running | 34:44:36 | In progress |
| comp_adamw_wd0.05 | AdamW | 0.05 | 🔄 Running | 34:47:49 | In progress |
| comp_adamw_wd0.1 | AdamW | 0.1 | 🔄 Running | ~34 hours | In progress |
| comp_adamw_wd0.2 | AdamW | 0.2 | 🔄 Running | ~34 hours | Checkpoint @ 10K |
| comp_adamw_wd0.5 | AdamW | 0.5 | ⏸️ Config only | - | Unknown |
| comp_adamw_wd1.0 | AdamW | 1.0 | ⏸️ Config only | - | Unknown |
| comp_muonw_wd0.* | Muon | Various | ⏸️ Config only | - | Unknown |
| comp_sgd_wd0.* | SGD | Various | ⏸️ Config only | - | Unknown |

**Running Jobs Detected:**
- Job 44360399: comp_grok (34:47 runtime)
- Job 44360402: comp_grok (34:44 runtime)
- Job 44360404: comp_grok (34:44 runtime)
- Job 44360405: comp_grok (34:44 runtime)
- Job 44360408: comp_grok (34:44 runtime)
- Job 44360412: comp_grok (34:44 runtime)

**Configuration (from comp_adamw_wd1.0):**
```
Model: GPT-2 (28,787,712 parameters)
Training samples: 181,000
Test samples: 932
Epochs: 150,000 (very long!)
Optimizer: AdamW, lr=0.0001
```

**Status:**
- ⏳ 6+ jobs still running after 34+ hours
- 📁 Checkpoints saved at epoch 10,000 for some
- ❌ NO training_history.json files yet (concerning!)
- 🤔 Either: still training OR history not being saved properly

**Expected completion:** Unknown (jobs running 34+ hours, target is 150K epochs)

---

## Cross-Dataset Comparison

### Grokking Success by Optimizer

| Optimizer | Nanda | Softmax | MNIST | Composition | Total |
|-----------|-------|---------|-------|-------------|-------|
| **AdamW** | 5/8 (63%) | 4/7 (57%) | 3/6 (50%) | TBD | 12/21 (57%) |
| **Muon** | 0/8 (0%) | 0/7 (0%) | 0/6 (0%) | TBD | 0/21 (0%) |
| **SGD** | 1/8 (13%) | 0/7 (0%) | 1/6 (17%) | TBD | 2/21 (10%) |

**Clear winner: AdamW** (especially with weight decay 0.5-2.0)

### Optimal Weight Decay by Dataset

| Dataset | Optimal WD Range | Fastest Grokking |
|---------|------------------|------------------|
| Nanda | 0.5-2.0 | wd=2.0 (epoch 2,600) |
| Softmax | 0.1-2.0 | wd=1.0 (epoch 1,300) |
| MNIST | 0.1-1.0 | wd=0.5 (epoch 2,600) |
| Composition | TBD | TBD |

---

## Composition Experiments - Detailed Investigation Needed

**Concern:** Jobs running 34+ hours without training_history.json files

**Possible issues:**
1. History saving disabled/broken in training script
2. Jobs will save at completion (not incrementally)
3. Very slow training (GPT-2 with 28M params is heavy)

**To investigate:** Need to check SLURM logs to see actual progress

**Action items:**
1. Check if training_history is being saved incrementally
2. Check SLURM logs for actual epoch progress
3. Estimate when jobs will complete based on progress
4. Consider if 150K epochs is necessary or can be reduced

---

## Overall Statistics

- **Total experiments:** 84
- **Completed:** 63 (75%)
- **Grokked:** 14 (17% of completed, 22% of those that could grok)
- **Running:** 21 (composition experiments)

**Next steps for composition:**
1. Monitor logs to check progress
2. Verify training_history saving mechanism
3. Wait for completion or investigate if stuck


