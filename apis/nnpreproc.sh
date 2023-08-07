# module load python

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip
pip install --no-index -e /scratch/$USER/TDSC-ABUS-2023/git-progs/nnUNet/

export nnUNet_raw_data_base=/scratch/$USER/TDSC-ABUS-2023/apis/
export nnUNet_preprocessed=/scratch/$USER/TDSC-ABUS-2023/apis/preprocessed/
export RESULTS_FOLDER=/scratch/$USER/TDSC-ABUS-2023/apis/trained_models/

nnUNet2_plan_and_preprocess -t "$1" --verify_dataset_integrity #-pl3d None
