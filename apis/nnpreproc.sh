# module load python

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip
pip install --no-index -e /scratch/$USER/git-progs/nnUNet/ # set this to your nnUNet folder

# Set the following path according to your system
export nnUNet_raw_data_base=/scratch/$USER/2023apis/
export nnUNet_preprocessed=/scratch/$USER/2023apis/preprocessed/
export RESULTS_FOLDER=/scratch/$USER/2023apis/trained_models/

nnUNet_plan_and_preprocess -t "$1" --verify_dataset_integrity #-pl3d None
