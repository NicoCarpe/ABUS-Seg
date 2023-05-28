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

export OUTPUT_FOLDER_MODEL1=/home/$USER/scratch/2023apis/output/20220720_1a/
export OUTPUT_FOLDER_MODEL2=/home/$USER/scratch/2023apis/output/20220720_1b/

export OUTPUT_FOLDER=/home/$USER/scratch/2023apis/output/20220720_t1/
export FOLDER_WITH_TEST_CASES=/home/$USER/scratch/2023apis/nnUNet_raw_data/Task301_A/imagesTs/


nnUNet_predict -i $FOLDER_WITH_TEST_CASES -o $OUTPUT_FOLDER_MODEL1 -tr nnUNetTrainerV2 -ctr nnUNetTrainerV2CascadeFullRes -m 2d -p nnUNetPlansv2.1 -t Task301_A -z
nnUNet_predict -i $FOLDER_WITH_TEST_CASES -o $OUTPUT_FOLDER_MODEL2 -tr nnUNetTrainerV2 -ctr nnUNetTrainerV2CascadeFullRes -m 3d_fullres -p nnUNetPlansv2.1 -t Task301_A -z
nnUNet_ensemble -f $OUTPUT_FOLDER_MODEL1 $OUTPUT_FOLDER_MODEL2 -o $OUTPUT_FOLDER -pp /scratch/$USER/2023apis/trained_models/nnUNet/ensembles/Task301_A/ensemble_2d__nnUNetTrainerV2__nnUNetPlansv2.1--3d_fullres__nnUNetTrainerV2__nnUNetPlansv2.1/postprocessing.json
