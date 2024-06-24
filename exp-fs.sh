#!/bin/bash
#SBATCH --partition=funky
#SBATCH --nodelist=edwards
#SBATCH --job-name=fs
#SBATCH --nodes=1
#SBATCH --time=7200
#SBATCH --gpus-per-node=1
#SBATCH --output=fs.out
#SBATCH --error=fs.err

dataset=$1
pi=$2
dry_run=${3:-False}

if [ "$pi" = 1 ]; then
  srun python3 p1.py --dataset_name $dataset --prompt_type few-shot --model_name meta-llama/Meta-Llama-3-8B-Instruct --dry_run $dry_run
elif [ "$pi" = 2 ]; then
  srun python3 p2.py --dataset_name $dataset --prompt_type few-shot --model_name meta-llama/Meta-Llama-3-8B-Instruct --dry_run $dry_run
fi
#meta-llama/Llama-2-13b-chat-hf
#meta-llama/Llama-2-7b-chat-hf
