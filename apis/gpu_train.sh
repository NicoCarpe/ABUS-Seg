#!/bin/bash
#SBATCH --account=def-punithak
#SBATCH --gres=gpu:v100l:1
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --mem=64G
#SBATCH --time=7:00:00
#SBATCH --mail-user=ngcarpen@ualberta.ca
#SBATCH --mail-type=ALL

sh nntrain.sh "$1" "$2" "$3"


## for i in {0..4}; do sbatch gpu_train.sh 501 2d $i; done
## for i in {0..4}; do sbatch gpu_train.sh 501 3d_lowres $i; done
## for i in {0..4}; do sbatch gpu_train.sh 501 3d_fullres $i; done
## for i in {0..4}; do sbatch gpu_train.sh 501 3d_cascade_fullres $i; done

