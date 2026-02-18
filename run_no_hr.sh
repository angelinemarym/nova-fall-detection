#!/bin/bash
#SBATCH --job-name=no_hr_train
#SBATCH --output=no_hr.out
#SBATCH --error=no_hr.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G            
#SBATCH --time=24:00:00      

source fall_detection_env/bin/activate

echo "Preprocessing No-HR Data..."
python preprocess_no_hr.py

echo "Starting 3-Axis (No HR)"
python main_no_hr.py --mode 3axis --output_dir ./results_no_hr/3axis

echo "Starting 6-Axis (No HR)"
python main_no_hr.py --mode 6axis --output_dir ./results_no_hr/6axis

echo "Starting 9-Axis (No HR)"
python main_no_hr.py --mode 9axis --output_dir ./results_no_hr/9axis