#!/bin/bash
#SBATCH --job-name=nnLandmark
#SBATCH --output=/work/tesi_averonese/nnLandmark/logs/nn_land_train%j.out
#SBATCH --error=/work/tesi_averonese/nnLandmark/logs/nn_land_train%j.err
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
DATASET_ID=001
FOLD=0

cd /homes/averonese/nnLandmark/

nnLM_train \
    ${DATASET_ID} \
    3d_fullres \
    ${FOLD} \
    -p nnUNetResEncUNetMPlans

echo "Training completed successfully!"