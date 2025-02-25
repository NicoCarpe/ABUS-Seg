#!/bin/bash -l
#SBATCH -J abus_train
#SBATCH --time=3-12:00:00
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

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1

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

nnUNetv2_train "$1" "$2" "$3" -tr nnUNetTrainer --npz ## Use this to start training from scratch
# nnUNetv2_train --c  "$1" "$2" "$3" --npz ##Use this to continue training

## Submit one job for all folds:
## sbatch run_train.sh 501 2d all
## sbatch run_train.sh 501 3d_fullres all

## Submit one job for each fold:
## for i in {0..4}; do sbatch run_train.sh 501 2d $i; done
## for i in {0..4}; do sbatch run_train.sh 501 3d_fullres $i; done