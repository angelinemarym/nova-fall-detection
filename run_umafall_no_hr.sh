#!/bin/bash
#SBATCH --job-name=no_hr_train
#SBATCH --output=umafall_out.out
#SBATCH --error=umafall_err.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00      

source fall_detection_env/bin/activate

echo "Preprocessing UMAFall No-HR Data..."
python preprocess_umafall_no_hr.py

echo "Starting 3-Axis UMAFall (No HR)"
python main_no_hr.py --mode 3axis --data_dir ./processed_tensors_umafall_no_hr --output_dir ./results_umafall_no_hr/3axis

echo "Starting 6-Axis UMAFall (No HR)"
python main_no_hr.py --mode 6axis --data_dir ./processed_tensors_umafall_no_hr --output_dir ./results_umafall_no_hr/6axis

echo "Starting 9-Axis UMAFall (No HR)"
python main_no_hr.py --mode 9axis --data_dir ./processed_tensors_umafall_no_hr --output_dir ./results_umafall_no_hr/9axis
