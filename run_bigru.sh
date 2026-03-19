#!/bin/bash
#SBATCH --job-name=bigru_experiments
#SBATCH --output=bigru.out
#SBATCH --error=bigru.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# 1. Activate your virtual environment
source fall_detection_env/bin/activate

echo "========================================="
echo "Tuning CNN-BiGRU (Bidirectional)"
echo "========================================="
python3 -u tune_cnn_gru.py --dataset fall_detection --mode 9axis --bidirectional
python3 -u tune_cnn_gru.py --dataset umafall --mode 9axis --bidirectional

echo "========================================="
echo "Running BiGRU-Only and CNN-BiGRU Experiments"
echo "========================================="

echo "Running BiGRU_Only for Fall Detection Data..."
python3 -u dl_experiments.py --model BiGRU_Only

echo "Running BiGRU_Only for UMAFall Dataset..."
python3 -u dl_experiments_umafall.py --model BiGRU_Only

echo "Running CNN_BiGRU for Fall Detection Data..."
python3 -u dl_experiments.py --model CNN_BiGRU

echo "Running CNN_BiGRU for UMAFall Dataset..."
python3 -u dl_experiments_umafall.py --model CNN_BiGRU

echo "========================================="
echo "BiGRU-based Experiments Completed."
echo "========================================="
