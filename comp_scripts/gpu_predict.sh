#!/bin/bash
#SBATCH --account=def-punithak
#SBATCH --gpus-per-node=v100l:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --mail-user=$USER@ualberta.ca
#SBATCH --mail-type=ALL

sh nnpredict.sh

