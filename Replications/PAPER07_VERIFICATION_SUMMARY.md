# Paper 7 Verification Summary: The Slingshot Mechanism

**Date:** November 20, 2025  
**Paper:** Thilak et al. (2022) - arXiv:2206.04817  
**Question:** Is the paper reproduced exactly? Do we achieve grokking in the same capacity?

---

## Executive Summary

### Answer: ⚠️ **PARTIAL REPLICATION WITH CRITICAL DEVIATION**

**Grokking Achievement:** ✅ **YES** - Spectacular cyclic grokking confirmed  
**Exact Reproduction:** ❌ **NO** - Critical deviation in weight decay configuration  
**Mechanism Verification:** ⚠️ **UNCERTAIN** - Slingshot mechanism not clearly demonstrated

---

## What Was Successfully Replicated

### ✅ Architecture (MATCHES)
- 2-layer decoder-only Transformer
- d_model=128, n_heads=4, d_mlp=512
- Standard softmax attention, no LayerNorm
- ~296,000 parameters

### ✅ Dataset (MATCHES)
- Modular addition task: (a + b) mod 97
- 50% train split (4,704 train, 4,705 test)
- Full batch gradient descent
- 100,000 epochs training

### ✅ Grokking Phenomenon (SPECTACULAR!)
- **221 major test accuracy jumps** (>20%)
- **Largest jump: 90.7%** at epoch 31,300
- **Extreme cyclic behavior:** Test accuracy oscillates 10-99%
- **Multiple grokking events:** Continuous oscillations throughout training
- **Final performance:** 98.1% train, 95.7% test

---

## Critical Deviation Identified

### ❌ Weight Decay Mismatch

**Paper's Main Claim:**
> "Without explicit regularization, grokking almost exclusively occurs at the onset of Slingshots"

The paper emphasizes that the Slingshot Mechanism occurs **WITHOUT weight decay** (i.e., weight_decay=0.0), demonstrating that grokking is an optimizer phenomenon independent of regularization.

**Our Implementation:**
- Used **weight_decay=1.0** (strong regularization)
- This is a **fundamental deviation** from the paper's core contribution

**Implications:**
1. We achieved grokking, but potentially through **regularization** (like Papers 1, 3) rather than pure optimizer dynamics
2. Cannot verify the paper's key claim about Slingshot occurring without regularization
3. The cyclic behavior might be a combination of:
   - Slingshot mechanism (optimizer dynamics)
   - Weight decay effects (regularization)
   - Interaction between both

---

## Slingshot Mechanism Analysis

### What We Analyzed

Conducted comprehensive analysis of last layer weight norms to verify the Slingshot Mechanism:

**Last Layer Norm Behavior:**
- Range: 1.70 to 5.34 (variation of 3.64)
- Mean: 3.23, Standard deviation: 0.65
- **Weak cyclic pattern** detected (only 1 major peak/trough)

**Correlation Analysis:**
- Correlation between norm changes and test accuracy changes: **r = 0.210**
- **Moderate positive correlation** (not strong)

**Test Accuracy Behavior:**
- **221 major jumps** detected (>20% in 100 epochs)
- **Extreme cyclic oscillations** throughout all 100,000 epochs
- Most dramatic grokking behavior across all papers

### ⚠️ Verdict: Mechanism Unclear

**Expected Slingshot Behavior:**
- Strong cyclic patterns in last layer weight norm
- Grokking occurs at onset of norm increases
- High correlation between norm cycles and accuracy

**Observed Behavior:**
- Weak norm cycles (not strongly periodic)
- Extreme accuracy cycles (highly periodic)
- Moderate correlation (r=0.210)

**Interpretation:**
The **cyclic grokking is confirmed**, but the **Slingshot mechanism is not clearly demonstrated**. The behavior could be:
1. Regularization-induced grokking (weight_decay=1.0 effect)
2. Slingshot mechanism obscured by weight decay
3. Hybrid of both mechanisms

---

## Do We Achieve Grokking "In the Same Capacity"?

### YES, but with Important Caveats

**Grokking Behavior: ✅ EXCEEDS PAPER**
- We observe **more dramatic** cyclic grokking than most papers
- 221 major jumps vs typical 1-10 in other papers
- Largest 90.7% jump is **extreme** by any standard
- Test accuracy oscillations (10-99%) demonstrate strong instability

**Cyclic Pattern: ✅ CONFIRMED**
- Test accuracy shows clear, repeated cycles
- Multiple grokking events throughout training
- Non-monotonic learning (accuracy goes up and down)

**Slingshot Mechanism: ⚠️ UNCERTAIN**
- Weight norm cycles are **weak** compared to test accuracy cycles
- Correlation is **moderate** (r=0.210), not strong
- Cannot definitively verify Slingshot as the primary mechanism

**Configuration: ❌ DEVIATED**
- Used weight_decay=1.0 instead of 0.0
- This changes the fundamental nature of the experiment
- Cannot verify paper's claim about regularization-free grokking

---

## Comparison to Paper's Claims

| Claim | Status | Evidence |
|-------|--------|----------|
| Grokking occurs | ✅ Confirmed | 221 major jumps, 95.7% final test acc |
| Cyclic behavior | ✅ Confirmed | Extreme oscillations (10-99%) |
| Multiple grokking events | ✅ Confirmed | Continuous throughout 100K epochs |
| Last layer norm cycles | ⚠️ Weak | Only 1 major peak detected |
| Slingshot without regularization | ❌ Not tested | Used WD=1.0, not WD=0.0 |
| Optimizer dynamics cause grokking | ⚠️ Uncertain | Confounded by weight decay |

---

## Key Questions Remaining

### 1. What happens with weight_decay=0?

**Critical Question:** Does cyclic grokking still occur without weight decay?

- ✅ If YES: Validates paper's core claim
- ❌ If NO: Suggests our results are regularization-induced

**Recommendation:** Re-run experiment with weight_decay=0.0 to test paper's main claim.

### 2. Are the norm cycles meaningful?

**Finding:** Last layer norm varies (1.7-5.3) but doesn't show strong cyclic pattern.

**Possible explanations:**
1. Simple peak detection may miss subtle cycles
2. Weight decay suppresses norm oscillations
3. Paper's examples may be cherry-picked or from different configurations
4. Norm tracking implementation may differ from paper

### 3. What optimizer does paper actually use?

**Uncertainty:** Paper mentions both Adam and AdamW. We used AdamW.

**Impact:** AdamW decouples weight decay from gradients, which could affect Slingshot dynamics differently than Adam.

---

## Final Verdict

### Architecture & Dataset: ✅ **MATCH**
All specifications match paper's description for modular addition experiments.

### Grokking Phenomenon: ✅ **CONFIRMED & EXCEEDS**
We achieve spectacular cyclic grokking that matches and exceeds the dramatic behavior expected.

### Exact Reproduction: ❌ **NO - Critical Deviation**
Used weight_decay=1.0 instead of 0.0, contradicting paper's emphasis on regularization-free grokking.

### Slingshot Mechanism: ⚠️ **UNCERTAIN**
- Weak norm cycles detected (not strong as expected)
- Moderate correlation (r=0.210) between norm and accuracy
- Cannot definitively confirm Slingshot as primary mechanism
- Likely confounded by weight decay

### Overall Answer to "Is Paper Reproduced Exactly?"

**NO**, but we achieve the **grokking phenomenon** described in the paper.

**Reasoning:**
1. ❌ **Critical deviation:** weight_decay=1.0 vs 0.0
2. ⚠️ **Mechanism unclear:** Can't verify Slingshot as primary cause
3. ✅ **Phenomenon confirmed:** Extreme cyclic grokking achieved
4. ✅ **Results quality:** High final accuracy, dramatic oscillations

### Answer to "Do We Achieve Grokking in Same Capacity?"

**YES**, we achieve cyclic grokking that **equals or exceeds** the paper's description.

**Evidence:**
- 221 major jumps (>20%) - extraordinary frequency
- 90.7% largest jump - most extreme across all papers
- Continuous oscillations - persistent cyclic behavior
- High final accuracy - 95.7% test (comparable to paper)

**However:**
- The **mechanism** behind our grokking may differ (regularization vs optimizer dynamics)
- We cannot claim to validate the paper's **core theoretical contribution** about Slingshot
- We demonstrate the **phenomenon** but not necessarily the **cause**

---

## Recommendations

### Priority 1: Critical Experiment

**Re-run with weight_decay=0.0**
- Test paper's core claim about regularization-free grokking
- Compare cyclic behavior with WD=0 vs WD=1.0
- Determine if Slingshot mechanism is present without weight decay

### Priority 2: Enhanced Analysis

**Improve norm cycle detection**
- Try alternative peak detection methods
- Examine norm derivatives and second derivatives
- Look for longer-period cycles (100s-1000s of epochs)

**Analyze correlation at different time scales**
- Current r=0.210 is epoch-to-epoch
- Check correlation over longer windows
- Examine phase relationships (does norm lead or lag accuracy?)

### Priority 3: Verify Optimizer

**Clarify Adam vs AdamW**
- Review paper carefully for optimizer specification
- Try Adam if AdamW was incorrect
- Document any behavioral differences

---

## Conclusion

We have achieved **spectacular cyclic grokking** that demonstrates the dramatic phenomena described in the paper, but we **cannot confirm exact reproduction** due to:

1. **Critical deviation:** Used weight_decay=1.0 instead of 0.0
2. **Mechanism uncertainty:** Slingshot not clearly demonstrated as primary cause
3. **Limited scope:** Single configuration (p=97, seed=42, WD=1.0)

The results are **scientifically valuable** as they show extreme cyclic grokking, but they do **not validate the paper's core theoretical claim** about Slingshot occurring without regularization.

**Status:** ⚠️ **PARTIAL REPLICATION**
- ✅ Phenomenon: Confirmed and spectacular
- ❌ Mechanism: Not verified
- ❌ Exact specs: Critical deviation present

---

## Files Generated

1. **PAPER07_VERIFICATION_REPORT.md** - Complete 11-section technical report
2. **slingshot_mechanism_analysis.png** - 4-panel visualization of norm vs accuracy
3. **analyze_slingshot_mechanism.py** - Analysis script with detailed statistics

All files located in: `/Replications/07_thilak_et_al_2022_slingshot/`

