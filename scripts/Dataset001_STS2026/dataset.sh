#!/bin/bash
#SBATCH --job-name=nnLandmark
#SBATCH --output=/work/tesi_averonese/nnLandmark/logs/STS_dataset_%j.out
#SBATCH --error=/work/tesi_averonese/nnLandmark/logs/STS_dataset_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=70G
#SBATCH --gres=gpu:1
#SBATCH --account=tesi_averonese
#SBATCH --partition=all_usr_prod
#SBATCH --time=12:00:00

module unload python/3.11.11-gcc-11.4.0 
module load python/3.10.16-gcc-11.4.0

source /homes/averonese/nnLandmark/nnlandmark-venv/bin/activate

export nnLM_results=/work/grana_maxillo/averonese_STS2026/nnLM/nnLM_results
export nnLM_preprocessed=/work/grana_maxillo/averonese_STS2026/nnLM/nnLM_preprocessed
export nnLM_raw=/work/grana_maxillo/averonese_STS2026/nnLM/nnLM_raw

DATASET_DIR=/work/grana_maxillo/averonese_STS2026/MICCAI-Chllenge-STS26-Task2
DATASET_ID=001
JSON_PATH=/work/grana_maxillo/averonese_STS2026/MICCAI-Chllenge-STS26-Task2/landmarks.json
PATIENTS_PATH=/work/grana_maxillo/averonese_STS2026/MICCAI-Chllenge-STS26-Task2/patients.txt

cd /homes/averonese/nnLandmark/

python /homes/averonese/nnLandmark/nnlandmark/dataset_conversion/nnLandmark/Dataset001_STS2026/prepare_dataset_splits.py \
    -i ${DATASET_DIR} \
    -id ${DATASET_ID} \
    --landmarks_json ${JSON_PATH} \
    --patients_file ${PATIENTS_PATH}

echo "Dataset splits prepared successfully!"