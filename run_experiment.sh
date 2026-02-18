#!/bin/bash
#SBATCH --job-name=nova_fall_train
#SBATCH --output=train.out
#SBATCH --error=train.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G            
## SBATCH --gpus=1             
#SBATCH --time=24:00:00      

# Activate your virtual environment
source fall_detection_env/bin/activate

echo "========================================="
echo "Starting 3-Axis Experiment (Acc + HR)"
echo "========================================="
python main.py --mode 3axis --epochs 100 --output_dir ./results_3axis

echo "========================================="
echo "Starting 6-Axis Experiment (Acc + Gyro + HR)"
echo "========================================="
python main.py --mode 6axis --epochs 100 --output_dir ./results_6axis

echo "========================================="
echo "Starting 9-Axis Experiment (Acc + Gyro + Mag + HR)"
echo "========================================="
python main.py --mode 9axis --epochs 100 --output_dir ./results_9axis

echo "All Experiments Completed Successfully."