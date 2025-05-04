#!/bin/bash
#SBATCH --job-name=not_that_deep
#SBATCH --account=cil
#SBATCH --output=../template_example/logs/%j.out
#SBATCH --error=../template_example/logs/%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --nodes=1

set -e
ROOT_DIR="../"
mkdir -p ${ROOT_DIR}/data/monocular_depth_output/{predictions,results} ${ROOT_DIR}/logs
module load cuda/12.6.0
source ~/.bashrc
conda activate /cluster/courses/cil/envs/monocular_depth/
python ${ROOT_DIR}/main.py