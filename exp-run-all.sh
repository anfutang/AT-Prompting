#!/bin/sh

dataset=$1
pi=$2

sbatch exp-at-cot-zs.sh $dataset $pi
sbatch exp-at-cot-fs.sh $dataset $pi
sbatch exp-cot-zs.sh $dataset $pi
sbatch exp-cot-fs.sh $dataset $pi
sbatch exp-at-zs.sh $dataset $pi
sbatch exp-at-fs.sh $dataset $pi
sbatch exp-zs.sh $dataset $pi
sbatch exp-fs.sh $dataset $pi


