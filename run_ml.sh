#!/bin/bash
#SBATCH --job-name=nova_ml_experiments
#SBATCH --output=ml_exp.out
#SBATCH --error=ml_exp.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

# 1. Activate your virtual environment
# Ensure you have installed dependencies on the LOGIN NODE first:
# pip install xgboost joblib scikit-learn numpy pandas
source fall_detection_env/bin/activate

echo "Environment Activated: $(which python3)"
echo "Python Version: $(python3 --version)"

echo "Current Directory: $(pwd)"
echo "Listing files:"
ls -l

if [ ! -f "ml_experiments.py" ]; then
    echo "ERROR: ml_experiments.py not found in $(pwd)"
    exit 1
fi

# Run the experiment script (using python3 from the activated env)
echo "Running experiment script..."
python3 -u ml_experiments.py

echo "========================================="
echo "Experiments Completed Successfully."
echo "Results saved in ./results_ml"
echo "========================================="
