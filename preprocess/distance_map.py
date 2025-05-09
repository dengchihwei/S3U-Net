# -*- coding = utf-8 -*-
# @File Name : distance_map
# @Date : 2024/10/21 23:38
# @Author : zhiweideng
# @E-mail : zhiweide@usc.edu


import numpy as np
import SimpleITK as SiTk
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt, gaussian_filter, maximum_filter
from skimage.morphology import skeletonize_3d

path = '/Users/zhiweideng/Desktop/NICR/VesselAnalysis/SMILE/valid_set/validate_label/sub007.nii'
# path = '/Users/zhiweideng/Desktop/COSTA-Dataset-v1/labelsTr/LocH1/N1stH-006_LocH1.nii.gz'
gt_image = SiTk.ReadImage(path)
gt_data = SiTk.GetArrayFromImage(gt_image)

skeleton = skeletonize_3d(gt_data)
distance = distance_transform_edt(gt_data)
radius = skeleton * distance

skeleton_coords = np.argwhere(skeleton)
foreground_coords = np.argwhere(gt_data)

skeleton_tree = cKDTree(skeleton_coords)
_, nearest_skeleton_indices = skeleton_tree.query(foreground_coords)

skel_radius = distance[skeleton > 0]
mapped_radius = np.zeros_like(gt_data, dtype=float)

# Assign the radius values to the corresponding foreground voxels
mapped_radius[tuple(foreground_coords.T)] = skel_radius[nearest_skeleton_indices]
# mapped_radius = gaussian_filter(mapped_radius, sigma=0.5) * gt_data
mapped_radius = maximum_filter(mapped_radius, size=9) * gt_data
mapped_radius = (mapped_radius < 1.1) * (mapped_radius > 0.0) * 1.0

skel_image = SiTk.GetImageFromArray(skeleton)
dist_image = SiTk.GetImageFromArray(mapped_radius)

skel_image.CopyInformation(gt_image)
dist_image.CopyInformation(gt_image)

SiTk.WriteImage(skel_image, '/Users/zhiweideng/Desktop/skel.nii.gz')
SiTk.WriteImage(dist_image, '/Users/zhiweideng/Desktop/dist.nii.gz')
