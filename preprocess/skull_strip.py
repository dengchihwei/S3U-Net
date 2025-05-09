# -*- coding = utf-8 -*-
# @File Name : skull_strip
# @Date : 2024/7/15 12:07
# @Author : zhiweideng
# @E-mail : zhiweide@usc.edu


import os
from datetime import datetime


def sub_exp(exp_name, job_name, cmd_str):
    tmp_path = os.path.join('/ifshome/zdeng/tmp/', exp_name)
    if not os.path.exists(tmp_path):
        os.mkdir(tmp_path)
    curr_time = datetime.now().strftime('%Y%m%d%H%M')
    output_log = os.path.join(tmp_path, job_name + '_' + curr_time + '.o')
    error_log = os.path.join(tmp_path, job_name + '_' + curr_time + '.e')
    qsub_str = 'echo ' + cmd_str + ' | qsub -q compute7.q -N '
    qsub_str += job_name + ' -o ' + output_log + ' -e ' + error_log
    # print(qsub_str)
    os.system(qsub_str)


def synth_strip(dataset_name, subject_path, output_path, mask_path):
    # subject_id = subject_path.split('/')[-1].split('.')[0]
    subject_id = subject_path.split('/')[-3]
    # call synthstrip command using freesurfer
    dir_path = os.path.dirname(os.path.realpath(__file__))
    cmd_str = '{}/mri_synthstrip.sh {} {} {}'.format(dir_path, subject_path, output_path, mask_path)
    exp_name = '{}_SynthStrip'.format(dataset_name)
    job_name = exp_name + '_' + subject_id
    sub_exp(exp_name, job_name, cmd_str)


if __name__ == "__main__":
    data_dir = '/ifs/loni/faculty/shi/spectrum/zdeng/MSA_Data/VesselLearning/Datasets'
    _dataset_name, folder = 'LSA_7T', 'nifti'
    subject_folder = '{}/{}/{}'.format(data_dir, _dataset_name, folder)
    output_folder = '{}/{}/{}_masked'.format(data_dir, _dataset_name, folder)
    mask_folder = '{}/{}/{}_mask'.format(data_dir, _dataset_name, folder)

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(mask_folder, exist_ok=True)

    subjects = sorted(os.listdir(subject_folder))
    for subject in subjects:
        subject_name = subject.split('.')[0]
        _image_path = os.path.join(subject_folder, '{}.nii.gz'.format(subject_name))
        _output_path = os.path.join(output_folder, '{}_masked.nii.gz'.format(subject_name))
        _mask_path = os.path.join(mask_folder, '{}_mask.nii.gz'.format(subject_name))
        synth_strip(_dataset_name, _image_path, _output_path, _mask_path)
