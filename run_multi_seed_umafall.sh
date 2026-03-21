#!/bin/bash
#SBATCH --job-name=nova_seed_umafall
#SBATCH --output=multi_seed_umafall_%j.out
#SBATCH --error=multi_seed_umafall_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# ============================================================
# run_multi_seed_umafall.sh
# Re-runs UMAFall dataset experiments with 5 different random
# seeds: 42, 0, 1, 123, 2026
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
    echo "[UMAFall | seed=${SEED}] Running ML experiments..."
    python3 -u ml_experiments_umafall.py --seed ${SEED}
    if [ $? -ne 0 ]; then
        echo "ERROR: ml_experiments_umafall.py failed for seed=${SEED}. Continuing..."
    fi

    echo "[UMAFall | seed=${SEED}] Running DL experiments..."
    python3 -u dl_experiments_umafall.py --seed ${SEED}
    if [ $? -ne 0 ]; then
        echo "ERROR: dl_experiments_umafall.py failed for seed=${SEED}. Continuing..."
    fi

    echo ""
    echo "Seed ${SEED} completed."
    echo ""
done

echo "============================================="
echo " UMAFall dataset multi-seed experiments completed!"
echo "============================================="
echo ""
echo "Output files per seed:"
echo "  results_ml_umafall/summary_results_seed<N>.csv"
echo "  results_dl_umafall/dl_comparison_seed<N>.csv"
echo ""
