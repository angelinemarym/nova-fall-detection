#!/bin/bash
#SBATCH --job-name=data_prep
#SBATCH --output=prep.out
#SBATCH --error=prep.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

# Load modules if needed
# module load anaconda3/2023.03

source activate my_tf_env

echo "Starting optimized preprocessing..."
python preprocess_optimized.py
echo "Done."