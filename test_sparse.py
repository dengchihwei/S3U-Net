# -*- coding = utf-8 -*-
# @File Name : test_sparse
# @Date : 2024/9/16 11:18
# @Author : zhiweideng
# @E-mail : zhiweide@usc.edu


import os
import torch
import argparse
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from sklearn import metrics
from miseval import evaluate
from scipy.ndimage import gaussian_filter
from skimage.morphology import remove_small_objects, binary_erosion

from loss import flux_loss
from train import read_json, Trainer


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='VESSEL12')
parser.add_argument('--split', type=str, default='valid')
parser.add_argument('--config_file', type=str, default='./configs/vessel12/adaptive_lc.json')
parser.add_argument('--test_path', type=str, default='../tests/SMILE/valid-UNET_100-2024-08-11-00-125/flux/')


def calc_metrics(args, thresh):
    trainer = Trainer(config_file=args.config_file)
    data_loader = trainer.get_data_loader(train=(args.split == 'train'), shuffle=False)
    preds, gts, responses = [], [], []
    records, measures = {}, ['AUC', 'ACC', 'DSC', 'SENS', 'SPEC', 'PREC']
    for subject in tqdm(data_loader.dataset.subjects, ncols=80, ascii=True):
        subject_name = subject['subject_name']
        response_file = os.path.join(args.test_path, 'flux_{}.nii.gz'.format(subject_name))
        response = sitk.GetArrayFromImage(sitk.ReadImage(response_file))
        bin_res = (response > thresh) * 1.0

        # get the labels
        label = subject['label']
        for loc in label.keys():
            x, y, z = loc
            responses.append(response[x, y, z])
            preds.append(bin_res[x, y, z])
            gts.append(label[loc])

    # calculate metrics
    for m in measures:
        if m not in records.keys():
            records[m] = []
        if m == 'AUC':
            records[m].append(metrics.roc_auc_score(np.array(gts), np.array(responses)))
        else:
            records[m].append(evaluate(np.array(gts), np.array(preds), metric=m))

    # show the final averaged metrics
    print('Threshold: {}'.format(thresh))
    for k in records.keys():
        print('Mean of {}: {}'.format(k, np.mean(records[k])))


def gen_gt(args):
    trainer = Trainer(config_file=args.config_file)
    data_loader = trainer.get_data_loader(train=(args.split == 'train'), shuffle=False)

    for subject in tqdm(data_loader.dataset.subjects, ncols=80, ascii=True):
        subject_name = subject['subject_name']
        tp_data = np.zeros_like(subject['image'])[0]
        tn_data = np.zeros_like(subject['image'])[0]

        # get the labels
        label = subject['label']
        for loc in label.keys():
            x, y, z = loc
            curr_label = label[loc]
            if curr_label == 1:
                tp_data[x, y, z] = 1.0
            else:
                tn_data[x, y, z] = 1.0

        tp_image = sitk.GetImageFromArray(tp_data)
        tp_image.CopyInformation(subject['meta_data'])
        sitk.WriteImage(tp_image, os.path.join(
            '../../VesselLearning/Datasets/vessel12/test_gt/{}_TP.nii.gz'.format(subject_name)))

        tn_image = sitk.GetImageFromArray(tn_data)
        tn_image.CopyInformation(subject['meta_data'])
        sitk.WriteImage(tn_image, os.path.join(
            '../../VesselLearning/Datasets/vessel12/test_gt/{}_TN.nii.gz'.format(subject_name)))


arguments = parser.parse_args()
# calc_metrics(arguments, thresh=0.5)
gen_gt(arguments)
