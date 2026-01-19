#!/bin/bash
#SBATCH --job-name=notebook_analysis
#SBATCH --output=/om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/New_Explorations/ICML_Submission/Power_AGOP_Study/slurm_scripts/logs/notebook_analysis_%j.out
#SBATCH --error=/om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/New_Explorations/ICML_Submission/Power_AGOP_Study/slurm_scripts/logs/notebook_analysis_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=normal

# ============================================================================
# Run Jupyter Notebook Analysis
# ============================================================================
#
# Executes the analyze_power_agop.ipynb notebook to generate all figures
# and analysis outputs, including:
#   - Experiment 5: Contingency Analysis
#   - Experiment 1-4: Framework cells (executed but may need data)
#   - All comparative visualizations
#
# ============================================================================

# Activate conda environment
CONDA_ENV=/om/scratch/Mon/mabdel03/9.520/conda_envs/grok_exp
export PATH=$CONDA_ENV/bin:$PATH

# Base directory
BASE_DIR=/om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/New_Explorations/ICML_Submission/Power_AGOP_Study

# Create logs directory if needed
mkdir -p ${BASE_DIR}/slurm_scripts/logs

echo "============================================================================"
echo "Running Notebook Analysis"
echo "============================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Notebook: ${BASE_DIR}/analysis/analyze_power_agop.ipynb"
echo "============================================================================"

# Change to analysis directory
cd ${BASE_DIR}/analysis

# Run the notebook using jupyter nbconvert
# This executes all cells and saves the output back to the notebook
$CONDA_ENV/bin/jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    --ExecutePreprocessor.timeout=3600 \
    --ExecutePreprocessor.kernel_name=python3 \
    analyze_power_agop.ipynb

EXIT_CODE=$?

echo ""
echo "============================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Notebook execution completed successfully"
    echo "Figures saved to: ${BASE_DIR}/analysis/figures/"
else
    echo "Notebook execution FAILED with exit code: $EXIT_CODE"
fi
echo "============================================================================"

exit $EXIT_CODE
