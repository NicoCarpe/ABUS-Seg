#!/bin/bash -l
#SBATCH -J abus_preproc
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1        
#SBATCH --cpus-per-task=1                
#SBATCH --mem=32GB                                                 
#SBATCH --account=def-punithak
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ngcarpen@ualberta.ca
#SBATCH --output=slurm_logs/out/%x_%j.out
#SBATCH --error=slurm_logs/err/%x_%j.err

module purge
module load python/3.10

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip
pip install --no-index pynrrd nibabel numpy

python nrrd_to_nifti.py

# Deactivate the virtual environment
deactivate