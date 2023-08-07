import os
import glob
import numpy as np
import nibabel as nib
import tensorflow as tf

from n2v.models import N2VConfig, N2V
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator


def load_images(path):
    image_list = []
    base_dir = os.path.normpath('~/scratch/TDSC-ABUS-2023/apis/nnUNet_raw/Dataset501_BreastTumour_(noisy)/imagesTr')
    file_list = glob.glob(base_dir + "/*.nii.gz")

    # Note: Nifti images are already in float32 format so no conversion is needed
    for index in range(len(file_list)):
        image = nib.load(file_list[index]).get_fdata()  # dims X, Y, Z
        image = np.flip(image)                          # swap axes to the right format Z, Y, X
        image = np.expand_dims(image, axis = -1)        # add a dimension for the number of channels
        image_list.append(image)                        # add images to our list

    return file_list, image_list                        # return the file list to save the denoised images with the same names


def create_patch_splits(patch_size, patch_size_z, imgs, generator):
    
    # extract training patches and store them as "X_train"
    # patches created with no overlap to enable splitting into training and validation sets
    patch_shape = (patch_size_z, patch_size, patch_size)
    patches = generator.generate_patches_from_list(imgs, shape=patch_shape)

    # split into 80/20 for train/validation
    train_val_split = int(patches.shape[0] * 0.8)
    X_train = patches[:train_val_split]
    X_val = patches[train_val_split:]

    return X_train, X_val


def predict_denoised_images(model, imgs):
    pred_imgs = []

    # process the images with the trained denoiser.
    for img in imgs:
        pred_img = model.predict(img, axes='ZYXC')
        pred_img = pred_imgs[:,:,:,0]                   # only one channel therefor the dimensions are Z, Y, X
        pred_img = np.flip(pred_img)                    # return to the nifti friendly format of X, Y, Z
        pred_img = nib.Nifti1Image(pred_img, np.eye(4)) # transform np array to nifti image
        pred_imgs.append(pred_img)                      # add the predicted image to our list of new images
    
    return pred_imgs





base_dir = None
_, imgs = load_images(base_dir)

for img in imgs:
    # expand dims to be in the correct format for n2v neural network
    img = np.expand_dims(img, axis=0)  

# helps extract patches for training and validation
generator = N2V_DataGenerator()

patch_size = 64
patch_size_z = 8

X_train, X_val = create_patch_splits(patch_size, patch_size_z, imgs, generator)

train_batch = 32
config = N2VConfig( X_train, 
                    unet_kern_size = 3, 
                    unet_n_first = 64, 
                    unet_n_depth = 3, 
                    train_steps_per_epoch = int(X_train.shape[0]/train_batch), 
                    train_epochs = 5, 
                    train_loss = 'mse', 
                    batch_norm = True, 
                    train_batch_size = train_batch, 
                    n2v_perc_pix = 0.198, 
                    n2v_patch_shape = (patch_size_z, patch_size, patch_size), 
                    n2v_manipulator = 'uniform_withCP', 
                    n2v_neighborhood_radius = 5, 
                    single_net_per_channel = False
                    )


# identify the model
model_name = 'n2v_3D_ABUS'

# the base directory in which the model will be stored
basedir = 'models'

# create the network model
model = N2V(config, model_name, basedir = basedir)

# train the model (history can be used to plot the training loss)
history = model.train(X_train, X_val)

# load the images that we would like to denoise
file_list, imgs = load_images(base_dir)

# predict the denooised images
pred_imgs = predict_denoised_images(model, imgs)

# save the denoised images
for index in len(range(pred_imgs)):
    # don't include path in file name so choose range -21 to -1 to get the last 20 chars
    nib.save(pred_imgs[index], os.path.join('~/scratch/TDSC-ABUS-2023/apis/nnUNet_raw/Dataset501_BreastTumour/imagesTr',  file_list[index][-21:-1] )) 

    


    

     
