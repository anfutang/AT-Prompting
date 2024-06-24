#!/bin/sh

dataset=$1
pi=$2
dry_run=${3:-False}

sbatch exp-at-cot-zs.sh $dataset $pi $dry_run
sbatch exp-at-cot-fs.sh $dataset $pi $dry_run
sbatch exp-cot-zs.sh $dataset $pi $dry_run
sbatch exp-cot-fs.sh $dataset $pi $dry_run
sbatch exp-at-zs.sh $dataset $pi $dry_run
sbatch exp-at-fs.sh $dataset $pi $dry_run
sbatch exp-zs.sh $dataset $pi $dry_run
sbatch exp-fs.sh $dataset $pi $dry_run


