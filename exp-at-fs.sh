#!/bin/bash
#SBATCH --partition=funky
#SBATCH --nodelist=bernard
#SBATCH --job-name=at-fs
#SBATCH --nodes=1
#SBATCH --time=7200
#SBATCH --gpus-per-node=1
#SBATCH --output=at-fs.out
#SBATCH --error=at-fs.err

dataset=$1
pi=$2

if [ "$pi" = 1 ]; then
  srun python3 p1.py --dataset_name $dataset --prompt_type AT-few-shot --model_name meta-llama/Meta-Llama-3-8B-Instruct
elif [ "$pi" = 2 ]; then
  srun python3 p2.py --dataset_name $dataset --prompt_type AT-few-shot --model_name meta-llama/Meta-Llama-3-8B-Instruct 
fi
#meta-llama/Llama-2-13b-chat-hf
#meta-llama/Llama-2-7b-chat-hf
