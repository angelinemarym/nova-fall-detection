#!/bin/bash
#SBATCH --job-name=umafall_dl
#SBATCH --output=umafall_dl_results.out
#SBATCH --error=umafall_dl_results.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00      

# Activate virtual environment
source fall_detection_env/bin/activate

echo "Step 1: Preprocessing UMAFall Data..."
python preprocess_umafall_no_hr.py

echo "Step 2: Running Deep Learning Experiments..."
python dl_experiments_umafall.py

echo "All DL UMAFall experiments completed."
