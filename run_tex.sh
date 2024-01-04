#!/bin/env bash
#
#SBATCH -J facadenet
#SBATCH --output facade_net_%j.txt
#SBATCH -e facade_net_%j.err
#
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=14-00:00:00
#SBATCH --cpus-per-task=12

env | grep -E '^SLURM|^SBATCH|^CUDA|^CONDA' | sort

python -m experiments facades_vec_pc_lr train facadenet
