# -*- coding = utf-8 -*-
# @File Name : dicom_to_nifti
# @Date : 2024/10/19 18:45
# @Author : zhiweideng
# @E-mail : zhiweide@usc.edu


import os
import dicom2nifti
import dicom2nifti.settings as settings

settings. disable_validate_slice_increment()

dataset_folder = '/ifs/loni/faculty/shi/spectrum/zdeng/MSA_Data/VesselLearning/Datasets/LSA_7T'
os.makedirs(os.path.join(dataset_folder, 'nifti'), exist_ok=True)


folders = sorted(os.listdir(dataset_folder))
# for folder in folders:
#     print(folder)
#     curr_subject_folder = os.path.join(dataset_folder, folder)
#     if not os.path.isdir(curr_subject_folder) or folder == 'nifti':
#         continue
#     subject_id = folder
#     dicom_series = sorted(os.listdir(curr_subject_folder))
#     for dicom in dicom_series:
#         dicom_folder = os.path.join(curr_subject_folder, dicom)
#         if 'FL_TOF' in dicom and os.path.isdir(dicom_folder):
#             # print(curr_nifti_path, dicom_folder)
#             dicom2nifti.convert_directory(dicom_folder, curr_subject_folder)
#             # print('Saved at {}'.format(curr_nifti_path))


for folder in folders:
    curr_subject_folder = os.path.join(dataset_folder, folder)
    if not os.path.isdir(curr_subject_folder) or folder == 'nifti':
        continue
    subject_id = folder
    files = sorted(os.listdir(curr_subject_folder))
    nifti_folder = os.path.join(dataset_folder, 'nifti')
    new_file_path = os.path.join(nifti_folder, '{}.nii.gz'.format(subject_id))
    for file in files:
        file_path = os.path.join(curr_subject_folder, file)
        if 'nii.gz' in file:
            os.system('cp {} {}'.format(file_path, new_file_path))
