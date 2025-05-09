# -*- coding = utf-8 -*-
# @File Name : paper_figure
# @Date : 2024/11/4 01:39
# @Author : zhiweideng
# @E-mail : zhiweide@usc.edu


import SimpleITK as sitk
from skimage.morphology import remove_small_objects, binary_opening


# method, threshold = 'ours', 0.0825 # smile
# method, threshold = 'frangi', 0.48 # smile
# method, threshold = 'sato', 0.080 # smile
# gt_path = '/Users/zhiweideng/Desktop/TMI_src/res_fig/SMILE/sub007_label.nii.gz'.format(method)
# src_path = '/Users/zhiweideng/Desktop/TMI_src/res_fig/SMILE/flux_sub007_{}.nii.gz'.format(method)
# bin_path = '/Users/zhiweideng/Desktop/TMI_src/res_fig/SMILE/flux_sub007_bin_{}.nii.gz'.format(method)
# trp_path = '/Users/zhiweideng/Desktop/TMI_src/res_fig/SMILE/flux_sub007_trp_{}.nii.gz'.format(method)
# fap_path = '/Users/zhiweideng/Desktop/TMI_src/res_fig/SMILE/flux_sub007_fap_{}.nii.gz'.format(method)
# fan_path = '/Users/zhiweideng/Desktop/TMI_src/res_fig/SMILE/flux_sub007_fan_{}.nii.gz'.format(method)

# method, subject_id, threshold = 'ours', 'h1_subj_9', 0.0793
# method, subject_id, threshold = 'frangi', 'h1_subj_9', 0.046
# method, subject_id, threshold = 'sato', 'h1_subj_9', 0.047
method, subject_id, threshold = 'meijering', 'h1_subj_9', 0.45
gt_path = '/Users/zhiweideng/Desktop/TMI_src/res_fig/LSA/labels/label_{}.nii.gz'.format(subject_id)
src_path = '/Users/zhiweideng/Desktop/TMI_src/res_fig/LSA/{}/flux_{}.nii.gz'.format(method, subject_id)
bin_path = '/Users/zhiweideng/Desktop/TMI_src/res_fig/LSA/{}/flux_bin_{}.nii.gz'.format(method, subject_id)
trp_path = '/Users/zhiweideng/Desktop/TMI_src/res_fig/LSA/{}/flux_trp_{}.nii.gz'.format(method, subject_id)
fap_path = '/Users/zhiweideng/Desktop/TMI_src/res_fig/LSA/{}/flux_fap_{}.nii.gz'.format(method, subject_id)
fan_path = '/Users/zhiweideng/Desktop/TMI_src/res_fig/LSA/{}/flux_fan_{}.nii.gz'.format(method, subject_id)


label = sitk.GetArrayFromImage(sitk.ReadImage(gt_path))
src_image = sitk.ReadImage(src_path)
src_response = sitk.GetArrayFromImage(src_image)
bin_response = src_response >= threshold

bin_response = remove_small_objects(bin_response, min_size=20)
trp_response = bin_response * label
fap_response = bin_response * (1.0 - label)
# fap_response = binary_opening(fap_response)
fan_response = (1.0 - bin_response) * label


bin_image = sitk.GetImageFromArray(bin_response + 0.0)
trp_image = sitk.GetImageFromArray(trp_response + 0.0)
fap_image = sitk.GetImageFromArray(fap_response + 0.0)
fan_image = sitk.GetImageFromArray(fan_response + 0.0)

bin_image.CopyInformation(src_image)
trp_image.CopyInformation(src_image)
fap_image.CopyInformation(src_image)
fan_image.CopyInformation(src_image)

sitk.WriteImage(bin_image, bin_path)
sitk.WriteImage(trp_image, trp_path)
sitk.WriteImage(fap_image, fap_path)
sitk.WriteImage(fan_image, fan_path)
