#!/bin/bash -l
#SBATCH -J abus_eval
#SBATCH --time=0-08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1        
#SBATCH --gpus-per-node=v100l:1    
#SBATCH --cpus-per-task=6                
#SBATCH --mem=80GB                                                 
#SBATCH --account=def-punithak
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ngcarpen@ualberta.ca
#SBATCH --output=slurm_logs/out/%x_%j.out
#SBATCH --error=slurm_logs/err/%x_%j.err

# Navigate to project root
cd ..  

# Initialize directories
mkdir -p slurm_logs/out slurm_logs/err
mkdir -p outputs

module purge
module load python/3.10

# Set up virtual environment
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

# Install dependencies
pip install --no-index --upgrade pip
pip install --no-index pandas tqdm pynrrd nibabel numpy

# Run evaluation script
python evaluation/abus_eval.py \        
  --gt_path data/nnUNet_raw/Dataset501_BreastTumour/labelsTs/ \  
  --seg_path outputs/nnUNet_results/ \   
  --save_path outputs/results.csv \      
  --tolerance 5

deactivate