# -*- coding: utf-8 -*-
"""
Single-Label 3D Ultrasound Segmentation Evaluation
"""

import numpy as np
import nibabel as nb
import os
from collections import OrderedDict
import pandas as pd
from SurfaceDice import compute_surface_distances, compute_surface_dice_at_tolerance, compute_dice_coefficient
join = os.path.join
basename = os.path.basename
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gt_path', type=str, default='', help='Path to ground truth NIfTI files')
parser.add_argument('--seg_path', type=str, default='', help='Path to predicted segmentation NIfTI files')
parser.add_argument('--save_path', type=str, default='', help='Path to save evaluation results')
parser.add_argument('--tolerance', type=int, default=5, help='Tolerance for surface dice (in mm)')
args = parser.parse_args()

gt_path = args.gt_path
seg_path = args.seg_path
save_path = args.save_path
tolerance = args.tolerance

filenames = os.listdir(seg_path)
filenames = [x for x in filenames if x.endswith('.nii.gz')]
filenames = [x for x in filenames if os.path.exists(join(seg_path, x))]
filenames.sort()

# Initialize metrics dictionary
seg_metrics = OrderedDict()
seg_metrics['Name'] = []
seg_metrics['DSC'] = []
seg_metrics['NSD'] = []  # Normalized Surface Dice

def get_binary_mask(data):
    """Convert data to binary tumor mask (assuming tumor is label 1)"""
    return np.uint8(data == 1)

for name in tqdm(filenames):
    seg_metrics['Name'].append(name)
    
    # Load ground truth and prediction
    gt_nii = nb.load(join(gt_path, name))
    seg_nii = nb.load(join(seg_path, name))
    
    # Get voxel spacing for surface calculations
    case_spacing = gt_nii.header.get_zooms()
    
    # Convert to binary masks
    gt_data = get_binary_mask(gt_nii.get_fdata())
    seg_data = get_binary_mask(seg_nii.get_fdata())

    # Calculate DSC
    if np.sum(gt_data) == 0 and np.sum(seg_data) == 0:
        dsc = 1.0  # Both empty
    elif np.sum(gt_data) == 0 or np.sum(seg_data) == 0:
        dsc = 0.0  # One empty, other not
    else:
        dsc = compute_dice_coefficient(gt_data, seg_data)
    
    # Calculate NSD (Normalized Surface Dice)
    if np.sum(gt_data) == 0 or np.sum(seg_data) == 0:
        nsd = float(dsc)  # NSD == DSC when one is empty
    else:
        try:
            surface_distances = compute_surface_distances(gt_data, seg_data, case_spacing)
            nsd = compute_surface_dice_at_tolerance(surface_distances, tolerance)
        except Exception as e:
            print(f"Error calculating NSD for {name}: {str(e)}")
            nsd = 0.0

    seg_metrics['DSC'].append(round(dsc, 4))
    seg_metrics['NSD'].append(round(nsd, 4))

# Save results
dataframe = pd.DataFrame(seg_metrics)
dataframe.to_csv(save_path, index=False)

# Calculate averages
case_avg = dataframe.mean(axis=0, numeric_only=True)
print(20 * '>')
print(f'Average DSC: {case_avg["DSC"]:.4f}')
print(f'Average NSD ({tolerance}mm): {case_avg["NSD"]:.4f}')
print(20 * '<')