#!/bin/bash
#SBATCH --job-name=nnLandmark
#SBATCH --output=/work/tesi_averonese/nnLandmark/logs/MML02_%j.out
#SBATCH --error=/work/tesi_averonese/nnLandmark/logs/MML02_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --account=tesi_averonese
#SBATCH --partition=all_usr_prod
#SBATCH --time=1:00:00

module unload python/3.11.11-gcc-11.4.0 
module load python/3.10.16-gcc-11.4.0

source /homes/averonese/nnLandmark/nnlandmark-venv/bin/activate

export nnLM_results=/work/grana_maxillo/averonese_STS2026/nnLM/nnLM_results
export nnLM_preprocessed=/work/grana_maxillo/averonese_STS2026/nnLM/nnLM_preprocessed
export nnLM_raw=/work/grana_maxillo/averonese_STS2026/nnLM/nnLM_raw

DATASET_DIR=/work/grana_maxillo/averonese_STS2026/
DATASET_ID=737

cd /homes/averonese/nnLandmark/

python /homes/averonese/nnLandmark/nnlandmark/dataset_conversion/nnLandmark/Dataset737_DMGLD_LFC/02_label_json.py \
    --src /work/grana_maxillo/averonese_STS2026/LFC_dataset \
    --dst /work/grana_maxillo/averonese_STS2026/nnLM/nnLM_raw/Dataset737_DMGLD_LFC

echo "label_json finished"