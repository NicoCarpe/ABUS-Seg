#!/bin/sh
# module load python/3.7
# module load cuda

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip
pip install --no-index -e /scratch/$USER/git-progs/nnUNet/

export nnUNet_raw_data_base=/scratch/$USER/2023apis/
export nnUNet_preprocessed=/scratch/$USER/2023apis/preprocessed/
export RESULTS_FOLDER=/scratch/$USER/2023apis/trained_models/

nnUNet_train "$2" nnUNetTrainerV2 "$1" "$3" --npz ## Use this to start training from scartch
# nnUNet_train -c "$2" nnUNetTrainerV2 "$1" "$3" --npz ##Use this to continue training