# -*- coding = utf-8 -*-
# @File Name : cldice
# @Date : 2023/10/2 02:10
# @Author : zhiweideng
# @E-mail : zhiweide@usc.edu


import os
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize, skeletonize_3d


def cl_score(v, s):
    """[this function computes the skeleton volume overlap]

    Args:
        v ([bool]): [image]
        s ([bool]): [skeleton]

    Returns:
        [float]: [computed skeleton volume intersection]
    """
    return np.sum(v*s)/np.sum(s)


def clDice(v_p, v_l):
    """[this function computes the cldice metric]

    Args:
        v_p ([bool]): [predicted image]
        v_l ([bool]): [ground truth image]

    Returns:
        [float]: [cldice metric]
    """
    if len(v_p.shape) == 2:
        tprec = cl_score(v_p, skeletonize(v_l))
        tsens = cl_score(v_l, skeletonize(v_p))
    else:
        tprec = cl_score(v_p, skeletonize_3d(v_l))
        tsens = cl_score(v_l, skeletonize_3d(v_p))
    return 2 * tprec * tsens / (tprec+tsens)


gt_path = '/ifs/loni/faculty/shi/spectrum/zdeng/MSA_Data/VesselLearning/codes/tests/drive/drive_sup/sup_test_{}.png'
bi_path = '/ifs/loni/faculty/shi/spectrum/zdeng/MSA_Data/VesselLearning/Datasets/DRIVE/test/1st_manual/'
files = sorted(os.listdir(bi_path))
scores = []

for i in range(20):
    curr_gt_path = gt_path.format(i)
    curr_bi_path = os.path.join(bi_path, files[i])
    gt = np.array(Image.open(curr_gt_path).convert('L')) > 0.0
    bi = np.array(Image.open(curr_bi_path)) > 0.0
    scores.append(clDice(bi, gt))

print(np.mean(scores))
