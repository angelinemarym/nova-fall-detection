#!/bin/bash
#SBATCH --job-name=nova_seed_main
#SBATCH --output=multi_seed_main_%j.out
#SBATCH --error=multi_seed_main_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# ============================================================
# run_multi_seed_main.sh
# Re-runs Main dataset (MobiAct / SisFall) experiments with
# 5 different random seeds: 42, 0, 1, 123, 2026
# ============================================================

SEEDS=(42 0 1 123 2026)

# Activate virtual environment
source fall_detection_env/bin/activate

echo "Environment Activated: $(which python3)"
echo "Python Version: $(python3 --version)"
echo "Current Directory: $(pwd)"
echo ""

TOTAL=${#SEEDS[@]}
IDX=0

for SEED in "${SEEDS[@]}"; do
    IDX=$((IDX + 1))
    echo "============================================="
    echo " Seed ${IDX}/${TOTAL}: ${SEED}"
    echo "============================================="

    echo ""
    echo "[Main | seed=${SEED}] Running ML experiments..."
    python3 -u ml_experiments.py --seed ${SEED}
    if [ $? -ne 0 ]; then
        echo "ERROR: ml_experiments.py failed for seed=${SEED}. Continuing..."
    fi

    echo "[Main | seed=${SEED}] Running DL experiments..."
    python3 -u dl_experiments.py --seed ${SEED}
    if [ $? -ne 0 ]; then
        echo "ERROR: dl_experiments.py failed for seed=${SEED}. Continuing..."
    fi

    echo ""
    echo "Seed ${SEED} completed."
    echo ""
done

echo "============================================="
echo " Main dataset multi-seed experiments completed!"
echo "============================================="
