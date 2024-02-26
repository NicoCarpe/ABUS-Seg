#!/bin/sh
module load python/3.9
module load cuda/11.8.0

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip
pip install --no-index -e /scratch/$USER/TDSC-ABUS-2023/git-progs/nnUNet/
pip install adan-pytorch

export nnUNet_raw=/scratch/$USER/TDSC-ABUS-2023/apis/nnUNet_raw/
export nnUNet_preprocessed=/scratch/$USER/TDSC-ABUS-2023/apis/nnUNet_preprocessed/
export nnUNet_results=/scratch/$USER/TDSC-ABUS-2023/apis/nnUNet_results/

nnUNetv2_train "$1" "$2" "$3" --npz ## Use this to start training from scratch
# nnUNetv2_train --c  "$1" "$2" "$3" --npz ##Use this to continue training