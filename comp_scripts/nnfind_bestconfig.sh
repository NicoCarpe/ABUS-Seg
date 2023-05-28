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

nnUNet_find_best_configuration -m 2d 3d_fullres -t "$1"