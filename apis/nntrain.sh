#!/bin/sh
# module load python/3.9
# module load cuda

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip
pip install --no-index -e /scratch/$USER/TDSC-ABUS-2023/git-progs/nnUNet/

export nnUNet_raw_data_base=/scratch/$USER/TDSC-ABUS-2023/apis/
export nnUNet_preprocessed=/scratch/$USER/TDSC-ABUS-2023/apis/preprocessed/
export RESULTS_FOLDER=/scratch/$USER/TDSC-ABUS-2023/apis/trained_models/

nnUNet_train "$2" nnUNetTrainerV2_Loss_DiceTopK10Focal  "$1" "$3" --npz ## Use this to start training from scartch
# nnUNet_train --c "$2" nnUNetTrainerV2_Loss_DiceTopK10Focal "$1" "$3" --npz ##Use this to continue training