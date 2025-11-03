# 🚀 START HERE - GOP Analysis Framework

Welcome to the Gradient Outer Product Analysis Framework for investigating grokking!

## What is This?

This framework automatically tracks **gradient outer products (GOPs)** and comprehensive metrics during neural network training to help you understand the grokking phenomenon.

## 30-Second Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Test the framework (takes 1 minute)
python test_framework.py

# 3. Run smallest experiment (takes ~30 minutes)
cd experiments/09_levi_linear
python wrapped_train.py --config ../../configs/09_levi_linear.yaml

# 4. Visualize results
cd ../../analysis
python visualize_gop.py --results_dir ../results/09_levi_linear
```

## What You Get

For each experiment, the framework tracks **every epoch**:

### 📊 Standard Metrics
- Train/test loss and accuracy

### 🔢 GOP Matrices
- Full gradient outer product (G ⊗ G^T)
- Per-layer gradient outer products

### 📈 GOP Metrics
- All eigenvalues and top-k eigenvectors
- Trace, norms (Frobenius, spectral, nuclear)
- Rank, condition number, determinant
- Eigenvalue statistics

### 💾 Storage
- Compressed HDF5 format
- Efficient incremental saving
- Configurable precision and compression

## Which Experiment Should I Run First?

| Experiment | Difficulty | Time | Storage | Why Run It |
|-----------|-----------|------|---------|------------|
| **09 - Levi Linear** | ⭐ Easy | ~6h | ~16 GB | Smallest model, fastest, BEST FOR TESTING |
| 08 - Doshi Poly | ⭐⭐ Medium | ~48h | ~50 GB | Medium size, clear grokking |
| 03 - Nanda Progress | ⭐⭐ Medium | ~72h | ~160 GB | Mechanistic interpretability baseline |
| 07 - Thilak Slingshot | ⭐⭐⭐ Hard | ~96h | ~450 GB | Interesting optimizer dynamics |
| 06 - Humayun MNIST | ⭐⭐⭐ Hard | ~120h | ~1 TB | Beyond algorithmic data |

## Read Next

1. **For First-Time Users:** Read `GETTING_STARTED.md`
2. **For Detailed Usage:** Read `USAGE_GUIDE.md`
3. **For Implementation Details:** Read `IMPLEMENTATION_SUMMARY.md`
4. **For Framework Overview:** Read `README.md`

## Need Help?

1. Run tests: `python test_framework.py`
2. Check logs in `experiments/XX/logs/`
3. Review working examples in `experiments/03_nanda_progress/`
4. Use template: `framework/template_wrapper.py`

## Ready-to-Run Experiments

✅ **Fully Implemented (Just run them!):**
- 03_nanda_progress
- 06_humayun_deep
- 07_thilak_slingshot
- 08_doshi_polynomials
- 09_levi_linear

📝 **Template Provided (Need adaptation):**
- 01_power_et_al
- 02_liu_effective
- 04_wang_implicit
- 05_liu_omnigrok
- 10_minegishi_tickets

Use `framework/template_wrapper.py` and `configs/CONFIG_TEMPLATE.yaml` to create wrappers for these.

---

**Let's understand the geometry of grokking! 🎓**

