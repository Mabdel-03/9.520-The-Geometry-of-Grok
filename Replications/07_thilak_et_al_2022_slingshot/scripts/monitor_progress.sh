#!/bin/bash
# Monitor training progress for the exact replication

echo "=========================================="
echo "PAPER 7: Monitoring Exact Replication"
echo "=========================================="
echo ""

# Check job status
echo "1. SLURM JOB STATUS"
echo "------------------------------------------"
squeue -u $USER | grep "slingshot" || echo "No slingshot jobs found"
echo ""

# Check if training_history.json exists and show progress
if [ -f "../results/logs/training_history.json" ]; then
    echo "2. TRAINING PROGRESS"
    echo "------------------------------------------"
    
    # Get the last epoch
    LAST_EPOCH=$(python3 -c "import json; f=open('../results/logs/training_history.json'); d=json.load(f); print(d['epoch'][-1])")
    LAST_TEST=$(python3 -c "import json; f=open('../results/logs/training_history.json'); d=json.load(f); print(f\"{d['test_acc'][-1]*100:.2f}\")")
    LAST_NORM=$(python3 -c "import json; f=open('../results/logs/training_history.json'); d=json.load(f); print(f\"{d['last_layer_norm'][-1]:.2f}\")")
    
    echo "Last epoch: $LAST_EPOCH / 100,000"
    echo "Test accuracy: $LAST_TEST%"
    echo "Last layer norm: $LAST_NORM"
    
    PROGRESS=$(python3 -c "print(int($LAST_EPOCH / 100000.0 * 100))")
    echo "Progress: $PROGRESS%"
    echo ""
    
    # Check for grokking
    GROK_CHECK=$(python3 -c "import json; f=open('../results/logs/training_history.json'); d=json.load(f); import numpy as np; test=np.array(d['test_acc']); print('YES' if (test >= 0.9).any() else 'NO')")
    echo "Grokking (>90%) detected: $GROK_CHECK"
    
else
    echo "2. TRAINING PROGRESS"
    echo "------------------------------------------"
    echo "Training not started yet or file not created"
    echo ""
fi

# Check latest log file
echo "3. RECENT LOG OUTPUT"
echo "------------------------------------------"
LATEST_LOG=$(ls -t logs/slingshot_exact_*.out 2>/dev/null | head -1)
if [ -f "$LATEST_LOG" ]; then
    echo "Log file: $LATEST_LOG"
    echo ""
    tail -20 "$LATEST_LOG"
else
    echo "No log files found yet"
fi

echo ""
echo "=========================================="
echo "Run this script again to check progress"
echo "=========================================="

