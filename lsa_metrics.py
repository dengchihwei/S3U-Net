# -*- coding = utf-8 -*-
# @File Name : lsa_metrics
# @Date : 1/9/23 11:49 AM
# @Author : zhiweideng
# @E-mail : zhiweide@usc.edu


import argparse
import subprocess
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
from sklearn import metrics
from train import Trainer
import seg_metrics.seg_metrics as sg
from scipy.ndimage import gaussian_filter

parser = argparse.ArgumentParser()


def dice_score(v_threshed, gt):
    if v_threshed.size == 0:
        return np.nan
    num = (2 * v_threshed * gt).mean()
    den = v_threshed.mean() + gt.mean() + 1e-100
    return num / den


def auc(vesselness, gt):
    gt = gt.astype(int)
    vesselness = (vesselness - vesselness.min())/(vesselness.max() - vesselness.min())
    fpr, tpr, thresh = metrics.roc_curve(gt.reshape(-1), vesselness.reshape(-1), pos_label=1)
    return metrics.auc(fpr, tpr)


def specificity(v_threshed, gt):
    tn = ((v_threshed == 0) & (gt == 0)).mean()
    fp = ((v_threshed == 1) & (gt == 0)).mean()
    print("TN: {}".format(tn))
    return tn / (tn + fp)


def sensitivity(v_threshed, gt):
    tp = ((v_threshed == 1) & (gt == 1)).mean()
    fn = ((v_threshed == 0) & (gt == 1)).mean()
    print("FN: {}".format(fn))
    return tp / (tp + fn)


def get_best_dice_threshold(response, label, thresholds):
    best_thresh, best_dice = None, -1
    n = int(len(thresholds) / 5.0)
    for thresh in thresholds[::n]:
        bin_response = (response >= thresh) + 0.0
        curr_dice = dice_score(bin_response, label)
        if curr_dice > best_dice:
            best_thresh = thresh
            best_dice = curr_dice
    print("Got best dice {:.4f} at threshold {}".format(best_dice, best_thresh))
    return best_thresh


def get_metrics(arguments):
    dice_scores, hd95_scores, avd_scores, best_thresholds = [], [], [], []
    sens_scores, spec_scores, auc_scores = [], [], []

    trainer = Trainer('./configs/lsa/unet.json')
    data_loader = trainer.get_data_loader(train=False, shuffle=False)
    subject_ids = [subj['subject_name'] for subj in data_loader.dataset.subjects]

    # compute the best thresholds
    for sub_id in tqdm(subject_ids):
        response_file = '../tests/LSA/LCN_valid/flux/flux_{}.nii.gz'.format(sub_id)
        label_file = '../../VesselLearning/Datasets/DarkVessels/UnilateralData/{}/label_{}.nii'.format(sub_id, sub_id)
        print(response_file)
        response_image, label_image = sitk.ReadImage(response_file), sitk.ReadImage(label_file)
        response, label = sitk.GetArrayFromImage(response_image), sitk.GetArrayFromImage(label_image)
        _, _, thresholds = metrics.roc_curve(label.reshape(-1), response.reshape(-1), pos_label=1)
        curr_best_thresh = get_best_dice_threshold(response, label, thresholds)
        best_thresholds.append(curr_best_thresh)

    final_threshold = np.mean(best_thresholds)
    # final_threshold = 0.5
    print('Final Threshold is {}.'.format(final_threshold))

    # compute the dice scores
    for sub_id in tqdm(subject_ids):
        response_file = '../tests/LSA/LCN_valid/flux/flux_{}.nii.gz'.format(sub_id)
        binary_file = '../tests/LSA/LCN_valid/flux/bin_res_{}.nii.gz'.format(sub_id)
        label_file = '../../VesselLearning/Datasets/DarkVessels/UnilateralData/{}/label_{}.nii'.format(sub_id, sub_id)
        response_image, label_image = sitk.ReadImage(response_file), sitk.ReadImage(label_file)
        response, label = sitk.GetArrayFromImage(response_image), sitk.GetArrayFromImage(label_image)
        bin_response = (response >= final_threshold) + 0.0
        # write the binary results
        bin_image = sitk.GetImageFromArray(bin_response)
        bin_image.CopyInformation(response_image)
        sitk.WriteImage(bin_image, binary_file)
        # get metrics
        labels = [0, 1]
        metric_results = sg.write_metrics(labels=labels[1:],
                                          gdth_path=label_file,
                                          pred_path=binary_file,
                                          metrics=['dice', 'hd95'],
                                          verbose=False)
        curr_dice = metric_results[0]['dice'][0]
        curr_hd95 = metric_results[0]['hd95'][0]
        dice_scores.append(curr_dice)
        hd95_scores.append(curr_hd95)
        # get AVD metric
        result1 = subprocess.run(['./EvaluateSegmentation', label_file, binary_file], capture_output=True, text=True)
        curr_avd = float(result1.stdout[result1.stdout.find('AVGDIS')+9:result1.stdout.find('AVGDIS')+19])
        curr_sens = float(result1.stdout[result1.stdout.find('SNSVTY') + 9:result1.stdout.find('SNSVTY') + 16])
        curr_spec = float(result1.stdout[result1.stdout.find('SPCFTY') + 9:result1.stdout.find('SPCFTY') + 16])
        avd_scores.append(curr_avd)
        sens_scores.append(curr_sens)
        spec_scores.append(curr_spec)
        auc_scores.append(auc(response, label))

    print("Dice Mean: {:.5f}, Std: {:.5f}".format(np.mean(dice_scores), np.std(dice_scores)))
    print("HD95 Mean: {:.5f}, Std: {:.5f}".format(np.mean(hd95_scores), np.std(hd95_scores)))
    print("AVD Mean: {:.5f}, Std: {:.5f}".format(np.mean(avd_scores), np.std(avd_scores)))
    print("SENS Mean: {:.5f}, Std: {:.5f}".format(np.mean(sens_scores), np.std(sens_scores)))
    print("SPEC Mean: {:.5f}, Std: {:.5f}".format(np.mean(spec_scores), np.std(spec_scores)))
    print("AUC Mean: {:.5f}, Std: {:.5f}".format(np.mean(auc_scores), np.std(auc_scores)))


if __name__ == '__main__':
    args = parser.parse_args()
    get_metrics(args)
