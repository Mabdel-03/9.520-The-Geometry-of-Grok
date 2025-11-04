#!/bin/bash
#
# Check status of all grokking experiments
#

echo "========================================================================"
echo "Grokking Experiments Status Check"
echo "Time: $(date)"
echo "========================================================================"
echo ""

# Show currently running jobs
echo "=== Currently Running Jobs ==="
squeue -u mabdel03 --format="%.10i %.30j %.10P %.8T %.15M %.6D %12R" | grep "grok" | sort -k2
echo ""

# Check for recent completions
echo "=== Recent Completions (last 2 hours) ==="
sacct -u mabdel03 --starttime="now-2hours" --format=JobID,JobName,State,Elapsed,End | grep "grok_" | grep "COMPLETED\|FAILED" | grep -v "batch\|extern" | tail -15
echo ""

# Check which papers have training data
echo "=== Papers with Training History Files ==="
for i in {01..10}; do
    paper_dir=$(ls -d ${i}_* 2>/dev/null | head -1)
    if [ -n "$paper_dir" ]; then
        json_file="$paper_dir/logs/training_history.json"
        if [ -f "$json_file" ]; then
            epochs=$(python3 -c "import json; data=json.load(open('$json_file')); print(len(data['epoch']))" 2>/dev/null || echo "0")
            echo "  Paper $i: ✓ ($epochs checkpoints)"
        else
            echo "  Paper $i: ✗ (no data yet)"
        fi
    fi
done
echo ""

echo "========================================================================"
echo "To view detailed training progress:"
echo "  tail -f XX_paper_name/logs/*.out"
echo "  tail -f XX_paper_name/*.out"  
echo ""
echo "To generate plots: python plot_results.py"
echo "========================================================================"

