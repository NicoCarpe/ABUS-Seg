#!/bin/bash -l
#SBATCH -J abus_predict
#SBATCH --time=1-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=v100l:1            
#SBATCH --cpus-per-task=6                  
#SBATCH --mem=64GB                                                 
#SBATCH --account=def-punithak
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ngcarpen@ualberta.ca
#SBATCH --output=slurm_logs/out/%x_%j.out
#SBATCH --error=slurm_logs/err/%x_%j.err

module purge

module load gcc/12.3
module load python/3.10
module load cuda/12.2
module load opencv 

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip
pip install torch==2.5.1+computecanada
pip install torchvision==0.20.0+computecanada 
pip install causal-conv1d
pip install mamba-ssm

# since openCV is already installed on the system, we need to remove the dependency from the setup.py file
sed -i "s/'opencv-python'//g" /scratch/$USER/ABUS-Seg/umamba/setup.py
pip install --no-index -e /scratch/$USER/ABUS-Seg/umamba/

export nnUNet_raw=/scratch/$USER/ABUS-Seg/data/nnUNet_raw/
export nnUNet_preprocessed=/scratch/$USER/ABUS-Seg/outputs/nnUNet_preprocessed/
export nnUNet_results=/scratch/$USER/ABUS-Seg/outputs/nnUNet_results/

# If you want to use ensambling:
# export OUTPUT_FOLDER_MODEL1=/home/$USER/scratch/ABUS-Seg/outputs/output/ensamble1/
# export OUTPUT_FOLDER_MODEL2=/home/$USER/scratch/ABUS-Seg/outputs/output/ensamble2/

export OUTPUT_FOLDER=/home/$USER/scratch/ABUS-Seg/outputs/output/predictions/
export INPUT_FOLDER=/home/$USER/scratch/ABUS-Seg/data/nnUNet_raw/Dataset501_BreastTumour/imagesTs/

nnUNetv2_predict -i $INPUT_FOLDER -o $OUTPUT_FOLDER -d "$1" -c 3d_lowres -f all -tr nnUNetTrainer --disable_tta

# NOTE: not sure if ensemble is supported, but can use it to combine predictions from multiple models using this:

# nnUNetv2_predict -i $INPUT_FOLDER -o $OUTPUT_FOLDER_MODEL1 -tr nnUNetTrainer -ctr nnUNetTrainerV2CascadeFullRes -m 2d -p nnUNetPlansv2.1 -t Dataset501_BreastTumour -z
# nnUNetv2_predict -i $INPUT_FOLDER -o $OUTPUT_FOLDER_MODEL2 -tr nnUNetTrainer -ctr nnUNetTrainerV2CascadeFullRes -m 3d_fullres -p nnUNetPlansv2.1 -t Dataset501_BreastTumour -z
# nnUNetv2_ensemble -f $OUTPUT_FOLDER_MODEL1 $OUTPUT_FOLDER_MODEL2 -o $OUTPUT_FOLDER -pp /scratch/$USER/ABUS-Seg/apis/trained_models/nnUNet/ensembles/Dataset501_BreastTumour/ensemble_2d__nnUNetTrainerV2_Loss_DiceTopK10Focal__nnUNetPlansv2.1--3d_fullres__nnUNetTrainerV2_Loss_DiceTopK10Focal__nnUNetPlansv2.1/postprocessing.json
