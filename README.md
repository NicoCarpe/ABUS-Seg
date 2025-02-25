# Automated-3D-Breast-Ultrasound-Segmentation

## Project Description

This project focuses on the automatic tumour segmentation of 3D Breast Ultrasound images, aiming to provide an efficient and accurate method for assisting in the diagnosis of breast cancer. We used the U-Mamba model, based on the nnUNet framework, in this project for its ability to automatically configure itself to work optimally with a given dataset and its flexibility in comparing segmentation using either 2D slices or 3D volumetric data.

![Segmentation results.](results_image.png)

## Setup

This work was built around running on the slurm job scheduler so simply download the code and navigate to the scripts directory.

1. Download code: `git clone https://github.com/bowang-lab/U-Mamba`
2. `cd scripts`


## Model Training
U-Mamaba is built on the popular [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) framework, so please follow this [guideline](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md) to prepare the dataset.

### Preprocessing

Configure both nnpreproc.sh and run_preproc.sh to your environment and submit job using:

```bash
sbatch run_preproc.sh DATASET_ID
```

### Train model

Configure both nntrain.sh and run_train.sh to your environment. You may also specify another trainer in nntrain.sh (you can see available trainers in 'ABUS-Seg/umamba/nnunetv2/training/nnUNetTrainer').


Train all folds on together:

```bash
sbatch run_train.sh DATASET_ID CONFIGUATION all
```

Submit job for each individual fold:

```bash
for i in {0..4}; do sbatch run_train.sh DATASET_ID CONFIGUATION $i; done
```

> `CONFIGURATION` can be `2d` and `3d_fullres` for 2D and 3D models, respectively.

## Inference

Configure both nnpredict.sh and run_predict.sh to your environment. Make certain that both the trainer and configuration match those used your training.

Run inference using:

```bash
sbatch run_predict.sh DATASET_ID
```


## Acknowledgements

We thank the authors of [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) and [U-Mamba](https://github.com/bowang-lab/U-Mamba) for making their valuable code publicly available.