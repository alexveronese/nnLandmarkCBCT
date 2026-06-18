#!/bin/bash
#SBATCH --job-name=nnLandmark
#SBATCH --output=/work/tesi_averonese/nnLandmark/logs/nn_land_%j.out
#SBATCH --error=/work/tesi_averonese/nnLandmark/logs/nn_land_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=70G
#SBATCH --gres=gpu:1
#SBATCH --account=tesi_averonese
#SBATCH --partition=all_usr_prod
#SBATCH --time=24:00:00

source /homes/averonese/nnLandmark/nnlandmark-venv/bin/activate

export nnLM_results=/work/grana_maxillo/averonese_STS2026/nnLM/nnLM_results
export nnLM_preprocessed=/work/grana_maxillo/averonese_STS2026/nnLM/nnLM_preprocessed
export nnLM_raw=/work/grana_maxillo/averonese_STS2026/nnLM/nnLM_raw

DATASET_DIR=/work/grana_maxillo/averonese_STS2026/

cd /homes/averonese/nnLandmark/

# 4. Lancio del codice Python
python /nnlandmark/dataset_conversion/nnLandmark/Dataset001_STS2026/prepare_dataset_splits.py \
    -i ${DATASET_DIR} \
    -id 001

echo "Dataset splits prepared successfully!"
