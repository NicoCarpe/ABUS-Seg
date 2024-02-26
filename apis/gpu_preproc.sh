#!/bin/bash
#SBATCH --account=def-punithak
#SBATCH --gres=gpu:v100l:1
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=72G
#SBATCH --time=2:00:00
#SBATCH --mail-user=ngcarpen@ualberta.ca
#SBATCH --mail-type=ALL

sh nnpreproc.sh "$1" 