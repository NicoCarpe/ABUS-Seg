#!/usr/bin/env python3

import os
from glob import glob
import nrrd 
import nibabel as nib
import numpy as np


### Training Image Conversion ###

baseDir = os.path.normpath('/mnt/c/Users/ncarp/Documents/code/TDSC-ABUS-2023/datasets/train_data/Data')

files = glob(baseDir + '/*.nrrd')

for file in files:
  print( file[-8:-5] )

  #load nrrd 
  _nrrd = nrrd.read(file)
  data = _nrrd[0]

  # we will exclude the headers for training
  header = _nrrd[1]

  #save nifti
  img = nib.Nifti1Image(data, np.eye(4))
  nib.save(img,os.path.join('/mnt/c/Users/ncarp/Documents/code/TDSC-ABUS-2023/datasets/nnUNet_raw/Dataset501_BreastTumour/imagesTr',  'ABUS_' + file[-8:-5] + '_0000' + '.nii.gz'))




### Training Lable Conversion ###

baseDir = os.path.normpath('/mnt/c/Users/ncarp/Documents/code/TDSC-ABUS-2023/datasets/train_data/MASK')

files = glob(baseDir + '/*.nrrd')

for file in files:
  print( file[-8:-5] )

  #load nrrd 
  _nrrd = nrrd.read(file)
  data = _nrrd[0]

  # we will exclude the headers for training
  header = _nrrd[1]

  #save nifti
  img = nib.Nifti1Image(data, np.eye(4))
  nib.save(img,os.path.join('/mnt/c/Users/ncarp/Documents/code/TDSC-ABUS-2023/datasets/nnUNet_raw/Dataset501_BreastTumour/labelsTr',  'ABUS_' + file[-8:-5] + '.nii.gz'))
