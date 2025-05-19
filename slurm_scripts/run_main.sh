#!/bin/bash
#SBATCH --job-name=not_that_deep
#SBATCH --account=cil_jobs
#SBATCH --output=../logs/%j.out
#SBATCH --error=../logs/%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --nodes=1

set -e
ROOT_DIR="../"
#MODEL_NAME="intel_dpt_large"
MODEL_NAME="dpt_hybrid_midas"
MODEL_DIR="${ROOT_DIR}/models/${MODEL_NAME}"
OUTPUT_DIR="${MODEL_DIR}/output"
mkdir -p ${OUTPUT_DIR}/{predictions,results}

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH}"

module load cuda/12.6.0
source ~/.bashrc
conda activate /cluster/courses/cil/envs/monocular_depth/
export OUTPUT_DIR
python ${MODEL_DIR}/main.py