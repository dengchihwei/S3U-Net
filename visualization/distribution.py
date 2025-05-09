# -*- coding = utf-8 -*-
# @File Name : distribution
# @Date : 2024/12/2 06:27
# @Author : zhiweideng
# @E-mail : zhiweide@usc.edu


import os
import numpy as np
import SimpleITK as sitk
from scipy.stats import wasserstein_distance
from scipy.stats import ttest_ind


subjects = [21, 22, 23]
methods = ['sato', 'meijering', 'frangi', 'oof', 'model']
path = '/ifs/loni/faculty/shi/spectrum/zdeng/MSA_Data/VesselLearning/Datasets/vessel12/'


for method in methods:
    pos, neg = [], []
    for sid in subjects:
        if method == 'model' or method == 'oof':
            res_path = os.path.join(path, 'old_{}_output'.format(method),
                                    'old_{}_response_{}.nii.gz'.format(method, sid))
        else:
            res_path = os.path.join(path, 'new_{}_output'.format(method),
                                    'new_{}_response_{}.nii.gz'.format(method, sid))
        response = sitk.GetArrayFromImage(sitk.ReadImage(res_path))
        response = np.transpose(response, (2, 1, 0))
        response = np.clip(response, a_min=0.0, a_max=None)
        # response = (response - np.min(response)) / (np.max(response) - np.min(response))
        # response = response / response.max()
        print(response.shape)

        label_path = os.path.join(path, 'test', 'VESSEL12_{}'.format(sid), 'VESSEL12_{}_Annotations.csv'.format(sid))
        annot_lines = open(label_path, "r").readlines()
        for line in annot_lines:
            strs = line.split(',')
            x, y, z = int(strs[0]), int(strs[1]), int(strs[2])
            label, curr_res = int(strs[3][0]), response[x, y, z]
            if label == 1:
                pos.append(curr_res)
            elif label == 0:
                neg.append(curr_res)

    print(method, wasserstein_distance(pos, neg))
    print(method, np.mean(pos), np.var(pos), np.mean(neg))





