#!/bin/bash
# Monitoring Dashboard for Grokking Experiments
# Run this periodically to check status

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                  GROKKING EXPERIMENTS - MONITORING DASHBOARD                 ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo

# Job Status
echo "📊 JOB STATUS"
echo "─────────────────────────────────────────────────────────────────────────────"
TOTAL=$(squeue -u $USER | grep -E "nanda_gr|mnist_gr" | wc -l)
RUNNING=$(squeue -u $USER | grep -E "nanda_gr|mnist_gr" | grep " R " | wc -l)
PENDING=$(squeue -u $USER | grep -E "nanda_gr|mnist_gr" | grep " PD " | wc -l)

echo "  Total experiments: $TOTAL"
echo "  Running:           $RUNNING"
echo "  Pending:           $PENDING"
echo "  Completed:         $((42 - TOTAL))"
echo

# Job Details
if [ $TOTAL -gt 0 ]; then
    echo "📋 RUNNING JOBS"
    echo "─────────────────────────────────────────────────────────────────────────────"
    squeue -u $USER -o "  %.10i %.10P %.18j %.2t %.10M %.6D %R" | grep -E "nanda_gr|mnist_gr" | grep " R " | head -10
    echo
fi

# Results Status
echo "💾 RESULTS STATUS"
echo "─────────────────────────────────────────────────────────────────────────────"
NANDA_RESULTS=$(ls -1d /om/scratch/Tue/mabdel03/9.520/results/optimizer_experiments/paper03_nanda/nanda_* 2>/dev/null | wc -l)
MNIST_RESULTS=$(ls -1d /om/scratch/Tue/mabdel03/9.520/results/optimizer_experiments/paper05_omnigrok/mnist_* 2>/dev/null | wc -l)

echo "  Nanda results:  $NANDA_RESULTS / 24"
echo "  MNIST results:  $MNIST_RESULTS / 18"
echo "  Total results:  $((NANDA_RESULTS + MNIST_RESULTS)) / 42"
echo

# Storage Usage
echo "💽 STORAGE USAGE"
echo "─────────────────────────────────────────────────────────────────────────────"
SCRATCH_SIZE=$(du -sh /om/scratch/Tue/mabdel03/9.520/ 2>/dev/null | cut -f1)
echo "  Scratch space:  $SCRATCH_SIZE (unlimited quota)"
echo

# Recent Activity
echo "📝 RECENT ACTIVITY"
echo "─────────────────────────────────────────────────────────────────────────────"
echo "  Latest output files:"
ls -1t /om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/New_Explorations/optimizer_experiments/slurm_scripts/logs/*.out 2>/dev/null | head -3 | while read f; do
    TIMESTAMP=$(stat -c %y "$f" | cut -d'.' -f1)
    BASENAME=$(basename "$f")
    echo "    $BASENAME  ($TIMESTAMP)"
done
echo

# Sample Progress
echo "🎯 SAMPLE PROGRESS"
echo "─────────────────────────────────────────────────────────────────────────────"
LATEST_OUT=$(ls -1t /om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/New_Explorations/optimizer_experiments/slurm_scripts/logs/nanda_*.out 2>/dev/null | head -1)
if [ -f "$LATEST_OUT" ]; then
    echo "  Latest Nanda job:"
    tail -10 "$LATEST_OUT" | grep -E "Epoch|Training:" | tail -3 | sed 's/^/    /'
fi
echo

# Completion Estimate
if [ $RUNNING -gt 0 ]; then
    echo "⏱️  ESTIMATED COMPLETION"
    echo "─────────────────────────────────────────────────────────────────────────────"
    echo "  With current resources:"
    echo "    - Nanda experiments: 2-5 days"
    echo "    - MNIST experiments: 4-7 days"
    echo "    - All experiments:   7-10 days"
fi

echo
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║  Monitor: watch -n 60 './MONITORING_DASHBOARD.sh'                           ║"
echo "║  Cancel:  scancel <JOB_ID>                                                   ║"
echo "║  Logs:    tail -f slurm_scripts/logs/nanda_*.out                             ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"

