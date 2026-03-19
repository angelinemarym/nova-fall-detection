#!/bin/bash
#SBATCH --job-name=hifd_experiments
#SBATCH --output=hifd_exp.out
#SBATCH --error=hifd_exp.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# 1. Activate virtual environment
source fall_detection_env/bin/activate

echo "========================================="
echo "Environment Diagnostics"
echo "========================================="
echo "Current Directory: $(pwd)"
echo "Python Path: $(which python3)"
echo "Python Version: $(python3 --version)"
echo "Dataset Structure (first 2 levels):"
ls -F hifd_dataset/ | head -n 20
ls -F hifd_dataset/subject_01/ | head -n 20
echo "========================================="



echo "Step 1: Preprocessing HIFD Dataset"
echo "========================================="
python3 -u hifd_preprocess.py
if [ $? -ne 0 ]; then
    echo "ERROR: Preprocessing failed. Exiting."
    exit 1
fi

echo "========================================="
echo "Step 2: Deep Learning Experiments (HIFD)"
echo "========================================="
python3 -u dl_experiments_hifd.py
if [ $? -ne 0 ]; then
    echo "ERROR: DL Experiments failed."
    exit 1
fi

echo "========================================="
echo "HIFD Pipeline Completed Successfully"
echo "========================================="

