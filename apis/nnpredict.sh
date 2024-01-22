#!/bin/sh
# module load python/3.7
# module load cuda

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip
pip install --no-index -e /scratch/$USER/git-progs/nnUNet/

export nnUNet_raw_data_base=/scratch/$USER/TDSC-ABUS-2023/apis/
export nnUNet_preprocessed=/scratch/$USER/TDSC-ABUS-2023/apis/preprocessed/
export RESULTS_FOLDER=/scratch/$USER/apis/TDSC-ABUS-2023/trained_models/

export OUTPUT_FOLDER_MODEL1=/home/$USER/scratch/TDSC-ABUS-2023/apis/output/20220720_1a/
export OUTPUT_FOLDER_MODEL2=/home/$USER/scratch/TDSC-ABUS-2023/apis/output/20220720_1b/

export OUTPUT_FOLDER=/home/$USER/scratch/TDSC-ABUS-2023/apis/output/20220720_t1/
export FOLDER_WITH_TEST_CASES=/home/$USER/scratch/TDSC-ABUS-2023/apis/nnUNet_raw_data/Dataset501_BreastTumour/imagesTs/


nnUNetv2_predict -i $FOLDER_WITH_TEST_CASES -o $OUTPUT_FOLDER_MODEL1 -tr nnUNetTrainerV2_Loss_DiceTopK10Focal -ctr nnUNetTrainerV2CascadeFullRes -m 2d -p nnUNetPlansv2.1 -t Dataset501_BreastTumour -z
nnUNetv2_predict -i $FOLDER_WITH_TEST_CASES -o $OUTPUT_FOLDER_MODEL2 -tr nnUNetTrainerV2_Loss_DiceTopK10Focal -ctr nnUNetTrainerV2CascadeFullRes -m 3d_fullres -p nnUNetPlansv2.1 -t Dataset501_BreastTumour -z
nnUNetv2_ensemble -f $OUTPUT_FOLDER_MODEL1 $OUTPUT_FOLDER_MODEL2 -o $OUTPUT_FOLDER -pp /scratch/$USER/TDSC-ABUS-2023/apis/trained_models/nnUNet/ensembles/Dataset501_BreastTumour/ensemble_2d__nnUNetTrainerV2_Loss_DiceTopK10Focal__nnUNetPlansv2.1--3d_fullres__nnUNetTrainerV2_Loss_DiceTopK10Focal__nnUNetPlansv2.1/postprocessing.json
