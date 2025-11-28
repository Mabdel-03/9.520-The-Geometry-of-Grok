#!/bin/bash

echo "Checking which experiments ran vs completed..."
echo ""

# Check Nanda
echo "NANDA EXPERIMENTS:"
echo "Expected: 24 (indices 0-23)"
echo "Ran (have logs): $(ls slurm_scripts/logs/nanda_agop_44382007_*.out 2>/dev/null | wc -l)"
echo "Completed (have results): $(find results/nanda -name "training_history.json" | wc -l)"
echo ""

# Check which Nanda indices failed
echo "Checking Nanda failures..."
for i in {0..23}; do
    LOG="slurm_scripts/logs/nanda_agop_44382007_$i.err"
    if [ -f "$LOG" ]; then
        if grep -q "CUDA error\|Traceback" "$LOG" 2>/dev/null; then
            ARCH=$(( i / 12 ))
            REM=$(( i % 12 ))
            OPT=$(( REM / 4 ))
            WD=$(( REM % 4 ))
            ARCHS=("mlp" "transformer")
            OPTS=("adamw" "muon" "sgd")
            WDS=("0.1" "1.0" "5.0" "10.0")
            echo "  Job $i FAILED: ${ARCHS[$ARCH]}_${OPTS[$OPT]}_wd${WDS[$WD]}"
        fi
    fi
done
echo ""

# Check Softmax  
echo "SOFTMAX EXPERIMENTS:"
echo "Expected: 24 (indices 0-23)"
echo "Ran (have logs): $(ls slurm_scripts/logs/softmax_agop_44382008_*.out 2>/dev/null | wc -l)"
echo "Completed (have results): $(find results/softmax -name "training_history.json" | wc -l)"
echo ""

# Check MNIST
echo "MNIST EXPERIMENTS:"
echo "Expected: 12"
echo "All failed due to CUDA incompatibility (GTX 1080 Ti)"
echo ""

