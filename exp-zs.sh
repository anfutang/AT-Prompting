#!/bin/bash
#SBATCH --partition=funky
#SBATCH --nodelist=chic
#SBATCH --job-name=zs
#SBATCH --nodes=1
#SBATCH --time=7200
#SBATCH --gpus-per-node=1
#SBATCH --output=zs.out
#SBATCH --error=zs.err

dataset=$1
pi=$2

if [ "$pi" = 1 ]; then
  srun python3 p1.py --dataset_name $dataset --prompt_type zero-shot --model_name meta-llama/Meta-Llama-3-8B-Instruct 
elif [ "$pi" = 2 ]; then
  srun python3 p2.py --dataset_name $dataset --prompt_type zero-shot --model_name meta-llama/Meta-Llama-3-8B-Instruct
fi
#meta-llama/Llama-2-13b-chat-hf
#meta-llama/Llama-2-7b-chat-hf
