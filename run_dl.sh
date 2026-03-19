#!/bin/bash
#SBATCH --job-name=nova_ml_experiments
#SBATCH --output=ml_exp.out
#SBATCH --error=ml_exp.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00

# 1. Activate your virtual environment
# Ensure you have installed dependencies on the LOGIN NODE first:
# pip install xgboost joblib scikit-learn numpy pandas
source fall_detection_env/bin/activate

echo "Environment Activated: $(which python3)"
echo "Python Version: $(python3 --version)"

echo "Current Directory: $(pwd)"
echo "Listing files:"
ls -l

# Run the experiment scripts
echo "Running Advanced Deep Learning experiments..."
python3 -u dl_experiments.py

echo "========================================="
echo "Experiments Completed Successfully."
echo "Results saved in ./results_dl"
echo "========================================="
