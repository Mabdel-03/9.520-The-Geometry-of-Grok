#!/bin/bash
#SBATCH --job-name=paper05_teacher_student
#SBATCH --output=../results/logs/teacher_student_%j.out
#SBATCH --error=../results/logs/teacher_student_%j.err
#SBATCH --time=6:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4

# Paper 05: Omnigrok - Teacher-Student Regression

mkdir -p ../results/logs

# Activate conda environment
source /om2/user/mabdel03/anaconda/etc/profile.d/conda.sh
conda activate /om2/user/mabdel03/conda_envs/SLT_Proj_Env

# Navigate to teacher-student grokking directory
cd /om2/user/mabdel03/files/Classes/9.520/9.520-The-Geometry-of-Grok/Replications/05_liu_et_al_2022_omnigrok/teacher-student/grokking || exit 1

echo "=========================================="
echo "Paper 05: Omnigrok - Teacher-Student"
echo "=========================================="
echo "Working directory: $(pwd)"
echo "Training student network to match teacher"
echo "Architecture: 3-layer MLP, width=100, Tanh"
echo "Optimizer: Adam (corrected)"
echo ""

# Run the teacher-student script
python teacher_student_grokking.py

echo ""
echo "Teacher-Student experiment complete!"

