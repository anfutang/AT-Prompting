#!/bin/bash
#SBATCH --partition=funky
#SBATCH --nodelist=rodgers
#SBATCH --job-name=aczs
#SBATCH --nodes=1
#SBATCH --time=7200
#SBATCH --gpus-per-node=1
#SBATCH --output=at-cot-zs.out
#SBATCH --error=at-cot-zs.err

dataset=$1
pi=$2
dry_run=${3:-False}

if [ "$pi" = 1 ]; then
  srun python3 p1.py --dataset_name $dataset --prompt_type AT-CoT-zero-shot --model_name meta-llama/Meta-Llama-3-8B-Instruct --dry_run $dry_run
elif [ "$pi" = 2 ]; then
  srun python3 p2.py --dataset_name $dataset --prompt_type AT-CoT-zero-shot --model_name meta-llama/Meta-Llama-3-8B-Instruct --dry_run $dry_run
fi

#meta-llama/Llama-2-13b-chat-hf
#meta-llama/Llama-2-13b-chat-hf
