# -*- coding = utf-8 -*-
# @File Name : gen_mip
# @Date : 2024/10/27 17:48
# @Author : zhiweideng
# @E-mail : zhiweide@usc.edu


import numpy as np
import SimpleITK as SiTk
from PIL import Image


src_path = '../../../VesselLearning/Datasets/SMILE/valid_set/validate_masked/sub007_masked.nii.gz'
dst_path = '../../../VesselLearning/Datasets/SMILE/valid_set/validate_masked/sub007_{}.png'

# src_path = '../../../VesselLearning/Datasets/DarkVessels/UnilateralData/h1_subj_2/T1SPC_NLM03_h1_subj_2.nii'
# dst_path = '../../../VesselLearning/Datasets/DarkVessels/UnilateralData/h1_subj_2/{}.png'


def normalize(image):
    max_val, min_val = image.max(), image.min()
    return (image - min_val) / (max_val - min_val)


tof_image = SiTk.GetArrayFromImage(SiTk.ReadImage(src_path))

# axi_mip_image_data = np.max(tof_image, axis=2).T
# sag_mip_image_data = np.max(tof_image, axis=0).T
# cor_mip_image_data = np.max(tof_image, axis=1).T
axi_mip_image_data = np.flip(np.max(tof_image, axis=0), axis=[0, 1])
sag_mip_image_data = np.flip(np.max(tof_image, axis=1), axis=[0, 1])
cor_mip_image_data = np.flip(np.max(tof_image, axis=2), axis=[0, 1])
axi_mip_image_data = normalize(axi_mip_image_data) * 255.0
sag_mip_image_data = normalize(sag_mip_image_data) * 255.0
cor_mip_image_data = normalize(cor_mip_image_data) * 255.0


axi_mip_image = Image.fromarray(axi_mip_image_data.astype(np.uint8), mode='L')
axi_mip_image.save(dst_path.format('AXI'))

sag_mip_image = Image.fromarray(sag_mip_image_data.astype(np.uint8), mode='L')
sag_mip_image = sag_mip_image.resize((sag_mip_image.size[0], sag_mip_image.size[1]))
sag_mip_image.save(dst_path.format('SAG'))

cor_mip_image = Image.fromarray(cor_mip_image_data.astype(np.uint8), mode='L')
cor_mip_image = cor_mip_image.resize((cor_mip_image.size[0], cor_mip_image.size[1]))
cor_mip_image.save(dst_path.format('COR'))
