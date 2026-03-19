#!/bin/bash
#SBATCH --job-name=umafall_all
#SBATCH --output=umafall_results.out
#SBATCH --error=umafall_results.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00      

# Activate virtual environment
source fall_detection_env/bin/activate

echo "Step 1: Preprocessing UMAFall Data..."
python preprocess_umafall_no_hr.py

echo "Step 2: Running Traditional ML Experiments..."
python ml_experiments_umafall.py

echo "Step 3: Running Deep Learning Experiments..."
python dl_experiments_umafall.py

echo "All UMAFall experiments completed."
