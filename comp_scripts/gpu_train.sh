#!/bin/bash
#SBATCH --account=def-punithak
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --mail-user=$USER@ualberta.ca
#SBATCH --mail-type=ALL

sh nntrain.sh "$1" "$2" "$3"


## for i in {0..4}; do sbatch gpu_train.sh 300 3d_fullres $i; done
## for i in {0..4}; do sbatch gpu_train.sh 300 2d $i; done

## for i in {0..4}; do sbatch gpu_train.sh 300 3d_lowres $i; done
## for i in {0..4}; do sbatch gpu_train.sh 300 3d_cascade_fullres $i; done

