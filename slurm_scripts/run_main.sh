#!/bin/bash
#SBATCH --job-name=not_that_deep
#SBATCH --account=cil_jobs
#SBATCH --output=../output/logs/%j.out
#SBATCH --error=../output/logs/%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --nodes=1

set -e
ROOT_DIR="../"
MODEL_OUTPUT_DIR="intel_dpt_large"
OUTPUT_DIR="${ROOT_DIR}/output/${MODEL_OUTPUT_DIR}"
mkdir -p ${ROOT_DIR}/output/${MODEL_OUTPUT_DIR}/{predictions,results}
module load cuda/12.6.0
source ~/.bashrc
conda activate /cluster/courses/cil/envs/monocular_depth/
export OUTPUT_DIR
python ${ROOT_DIR}/main.py