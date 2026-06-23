#!/bin/bash
#SBATCH --job-name=nnLandmark
#SBATCH --output=/work/tesi_averonese/nnLandmark/logs/MML04_%j.out
#SBATCH --error=/work/tesi_averonese/nnLandmark/logs/MML04_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --account=tesi_averonese
#SBATCH --partition=all_usr_prod
#SBATCH --time=1:00:00

source /homes/averonese/nnLandmark/nnlandmark-venv/bin/activate

export nnLM_results=/work/grana_maxillo/averonese_STS2026/nnLM/nnLM_results
export nnLM_preprocessed=/work/grana_maxillo/averonese_STS2026/nnLM/nnLM_preprocessed
export nnLM_raw=/work/grana_maxillo/averonese_STS2026/nnLM/nnLM_raw

DATASET_DIR=/work/grana_maxillo/averonese_STS2026/
DATASET_ID=773

cd /homes/averonese/nnLandmark/

python /homes/averonese/nnLandmark/nnlandmark/dataset_conversion/nnLandmark/Dataset733_MML/04_label_maps.py \
    --base /work/grana_maxillo/averonese_STS2026/mmld_dataset \
    --landmarks /work/grana_maxillo/averonese_STS2026/nnLM/nnLM_raw/Dataset733_MML/all_landmarks_voxel.json \
    --output /work/grana_maxillo/averonese_STS2026/nnLM/nnLM_raw/Dataset733_MML/labelsTr

echo "label_maps finished"