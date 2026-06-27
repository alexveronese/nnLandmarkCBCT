#!/bin/bash
#SBATCH --job-name=nnLandmark
#SBATCH --output=/work/tesi_averonese/nnLandmark/logs/LFC_predict%j.out
#SBATCH --error=/work/tesi_averonese/nnLandmark/logs/LFC_predict%j.err
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

export nnLM_results=/work/grana_maxillo/averonese_STS2026/nnLM/nnLM_results
export nnLM_preprocessed=/work/grana_maxillo/averonese_STS2026/nnLM/nnLM_preprocessed
export nnLM_raw=/work/grana_maxillo/averonese_STS2026/nnLM/nnLM_raw

DATASET_DIR=/work/grana_maxillo/averonese_STS2026/nnLM/nnLM_raw/Dataset737_DMGLD_LFC/imagesTs
DATASET_ID=737
OUTPUT_DIR=/work/grana_maxillo/averonese_STS2026/nnLM/predictions/Dataset737_DMGLD_LFC/

cd /homes/averonese/nnLandmark/

nnLM_evaluate \
    -d ${DATASET_ID} \
    -pred ${OUTPUT_DIR}

echo "Training completed successfully!"