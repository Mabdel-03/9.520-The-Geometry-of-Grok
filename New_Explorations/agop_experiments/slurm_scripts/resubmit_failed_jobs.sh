#!/bin/bash
# Resubmit failed WD sweep jobs with GPU constraint to exclude 1080 Ti nodes

cd /om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/New_Explorations/agop_experiments

echo "============================================================"
echo "Checking for failed experiments and resubmitting..."
echo "============================================================"

# Arrays for tracking failed job indices
MNIST_FAILED=()
NANDA_FAILED=()
SOFTMAX_FAILED=()

# MNIST WD Sweep - 27 jobs (3 optimizers x 9 weight decays)
OPTIMIZERS=("adamw" "muon" "sgd")
WEIGHT_DECAYS=("0" "0.01" "0.1" "0.5" "1.0" "5.0" "10.0" "50.0" "100.0")

echo ""
echo "=== MNIST WD Sweep ==="
for opt_idx in 0 1 2; do
    for wd_idx in 0 1 2 3 4 5 6 7 8; do
        job_idx=$((opt_idx * 9 + wd_idx))
        opt=${OPTIMIZERS[$opt_idx]}
        wd=${WEIGHT_DECAYS[$wd_idx]}
        exp_dir="results/mnist_wd_sweep/mnist_${opt}_wd${wd}_seed42"
        
        if [ ! -f "$exp_dir/training_history.json" ] || [ ! -s "$exp_dir/training_history.json" ]; then
            echo "  FAILED: $opt wd=$wd (job $job_idx)"
            MNIST_FAILED+=($job_idx)
        fi
    done
done
echo "  Total failed: ${#MNIST_FAILED[@]}/27"

# Nanda WD Sweep - 54 jobs (2 arch x 3 optimizers x 9 weight decays)
ARCHITECTURES=("mlp" "transformer")

echo ""
echo "=== Nanda WD Sweep ==="
for arch_idx in 0 1; do
    for opt_idx in 0 1 2; do
        for wd_idx in 0 1 2 3 4 5 6 7 8; do
            job_idx=$((arch_idx * 27 + opt_idx * 9 + wd_idx))
            arch=${ARCHITECTURES[$arch_idx]}
            opt=${OPTIMIZERS[$opt_idx]}
            wd=${WEIGHT_DECAYS[$wd_idx]}
            exp_dir="results/nanda_wd_sweep/nanda_${arch}_${opt}_wd${wd}_seed42"
            
            if [ ! -f "$exp_dir/training_history.json" ] || [ ! -s "$exp_dir/training_history.json" ]; then
                echo "  FAILED: $arch $opt wd=$wd (job $job_idx)"
                NANDA_FAILED+=($job_idx)
            fi
        done
    done
done
echo "  Total failed: ${#NANDA_FAILED[@]}/54"

# Softmax WD Sweep - 54 jobs (2 arch x 3 optimizers x 9 weight decays)
echo ""
echo "=== Softmax WD Sweep ==="
for arch_idx in 0 1; do
    for opt_idx in 0 1 2; do
        for wd_idx in 0 1 2 3 4 5 6 7 8; do
            job_idx=$((arch_idx * 27 + opt_idx * 9 + wd_idx))
            arch=${ARCHITECTURES[$arch_idx]}
            opt=${OPTIMIZERS[$opt_idx]}
            wd=${WEIGHT_DECAYS[$wd_idx]}
            exp_dir="results/softmax_wd_sweep/softmax_${arch}_${opt}_wd${wd}_seed42"
            
            if [ ! -f "$exp_dir/training_history.json" ] || [ ! -s "$exp_dir/training_history.json" ]; then
                echo "  FAILED: $arch $opt wd=$wd (job $job_idx)"
                SOFTMAX_FAILED+=($job_idx)
            fi
        done
    done
done
echo "  Total failed: ${#SOFTMAX_FAILED[@]}/54"

# Resubmit failed jobs
echo ""
echo "============================================================"
echo "Resubmitting failed jobs with GPU constraint (turing|ampere)"
echo "============================================================"

cd slurm_scripts

if [ ${#MNIST_FAILED[@]} -gt 0 ]; then
    # Convert array to comma-separated list
    MNIST_ARRAY=$(IFS=,; echo "${MNIST_FAILED[*]}")
    echo ""
    echo "Submitting MNIST jobs: $MNIST_ARRAY"
    sbatch --array=$MNIST_ARRAY run_mnist_wd_sweep.sh
fi

if [ ${#NANDA_FAILED[@]} -gt 0 ]; then
    NANDA_ARRAY=$(IFS=,; echo "${NANDA_FAILED[*]}")
    echo ""
    echo "Submitting Nanda jobs: $NANDA_ARRAY"
    sbatch --array=$NANDA_ARRAY run_nanda_wd_sweep.sh
fi

if [ ${#SOFTMAX_FAILED[@]} -gt 0 ]; then
    SOFTMAX_ARRAY=$(IFS=,; echo "${SOFTMAX_FAILED[*]}")
    echo ""
    echo "Submitting Softmax jobs: $SOFTMAX_ARRAY"
    sbatch --array=$SOFTMAX_ARRAY run_softmax_wd_sweep.sh
fi

echo ""
echo "Done! Check queue with: squeue -u \$USER"

