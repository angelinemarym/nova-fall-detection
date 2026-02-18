#!/bin/bash
#SBATCH --job-name=nova_hpo
#SBATCH --output=tuning.out
#SBATCH --error=tuning.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8      # Slightly more CPUs to speed up the loop
#SBATCH --mem=32G            
#SBATCH --time=24:00:00      

source fall_detection_env/bin/activate

echo "Starting Hyperparameter Optimization..."
python tune_hyperparameters.py --mode 9axis
echo "Tuning Completed."