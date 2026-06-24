#!/bin/bash
#SBATCH --job-name=nnLandmark
#SBATCH --output=/work/tesi_averonese/nnLandmark/logs/nn_land_train_fold0%j.out
#SBATCH --error=/work/tesi_averonese/nnLandmark/logs/nn_land_train_fold0%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=70G
#SBATCH --gres=gpu:1
#SBATCH --account=tesi_averonese
#SBATCH --partition=all_usr_prod
#SBATCH --time=24:00:00

module unload python/3.11.11-gcc-11.4.0 
module load python/3.10.16-gcc-11.4.0

source /homes/averonese/nnLandmark/nnlandmark-venv/bin/activate

export nnUNet_compile=False
export nnLM_results=/work/grana_maxillo/averonese_STS2026/nnLM/nnLM_results
export nnLM_preprocessed=/work/grana_maxillo/averonese_STS2026/nnLM/nnLM_preprocessed
export nnLM_raw=/work/grana_maxillo/averonese_STS2026/nnLM/nnLM_raw

DATASET_DIR=/work/grana_maxillo/averonese_STS2026/
DATASET_ID=733
FOLD=0

cd /homes/averonese/nnLandmark/

nnLM_train \
    ${DATASET_ID} \
    3d_fullres \
    ${FOLD} \
    -p nnUNetResEncUNetMPlans

echo "Training completed successfully!"