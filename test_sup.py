# -*- coding = utf-8 -*-
# @File Name : test_sup
# @Date : 2024/7/1 04:35
# @Author : zhiweideng
# @E-mail : zhiweide@usc.edu


import os
import torch
import argparse
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from train import read_json, Trainer

from sklearn import metrics
from miseval import evaluate
from scipy.ndimage import gaussian_filter
from skimage.morphology import remove_small_objects, binary_erosion


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='LSA')
parser.add_argument('--split', type=str, default='valid')
parser.add_argument('--method', type=str, default='FT_60')
parser.add_argument('--epoch', type=str, default='25')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--config_file', type=str, default='./configs/lsa/unet.json')
parser.add_argument('--model_path', type=str,
                    default='../trained_models/LSA/LSA_FT_60/2024-08-11-14/SUPERVISED-100-epoch.pth')


def model_output(args):
    # load checkpoint
    checkpoint = torch.load(args.model_path, map_location=args.device)
    try:
        assert checkpoint['configer']['arch']['type'] == read_json(args.config_file)['arch']['type']
        assert checkpoint['configer']['arch']['args'] == read_json(args.config_file)['arch']['args']
    except AssertionError:
        print(checkpoint['configer']['arch'])
        print(read_json(args.config_file)['arch'])
        return

    # create the trainer
    trainer = Trainer(config_file=args.config_file)

    # create the model and load parameters
    trainer.trainer_conf['gpu_device_num'] = 1
    model = trainer.get_model()
    model.load_state_dict(checkpoint['model'])
    model.to(args.device).eval()
    print('load model done')

    # get the data loaders
    trainer.dataset_conf['batch_size'] = 1
    if 'augment' in trainer.dataset_conf['args'].keys():
        trainer.dataset_conf['args']['augment'] = False
    data_loader = trainer.get_data_loader(train=(args.split == 'train'), shuffle=False)

    # start inference
    prev_image_id = -1
    res, res_list = None, []
    px, py = data_loader.dataset.patch_sizes[:2]
    pz = data_loader.dataset.patch_sizes[2] if len(data_loader.dataset.patch_sizes) == 3 else None
    for batch in tqdm(data_loader, ncols=80, ascii=True, desc='Inference: '):
        curr_image_id = batch['image_id']
        if curr_image_id != prev_image_id:
            if prev_image_id != -1:
                res_list.append(res.copy())
            image_shape = data_loader.dataset.subjects[curr_image_id]['image'].shape[1:]
            res = np.zeros(image_shape)
            prev_image_id = curr_image_id

        # model inference
        image_patch = batch['image'].to(args.device)
        gts = batch['label'].to(args.device)
        with torch.no_grad():
            curr_pred, _ = model(image_patch, gts)
            curr_pred = torch.sigmoid(curr_pred)

            # update the results
            b = curr_pred.shape[0]
            for i in range(b):
                x, y = batch['start_coord'][i][:2]
                if pz is not None:
                    z = batch['start_coord'][i][2]
                    res[x:x+px, y:y+py, z:z+pz] = np.maximum(res[x:x+px, y:y+py, z:z+pz], curr_pred.cpu().numpy())
                else:
                    res[x:x + px, y:y + py] = np.maximum(res[x:x + px, y:y + py], curr_pred.cpu().numpy())
    res_list.append(res.copy())
    print(res_list[0].shape)

    # save the results lists to files
    os.makedirs("../tests/{}/{}_{}".format(args.data, args.method, args.split), exist_ok=True)
    os.makedirs("../tests/{}/{}_{}/res".format(args.data, args.method, args.split), exist_ok=True)
    assert len(res_list) == len(data_loader.dataset.subjects)
    for i in range(len(res_list)):
        mask = data_loader.dataset.subjects[i]['mask'][0]
        subject_name = data_loader.dataset.subjects[i]['subject_name']
        if len(res_list[i].shape) == 3:
            res_path = '../tests/{}/{}_{}/res/res_{}.nii.gz'.format(args.data, args.method, args.split, subject_name)
            meta_data = data_loader.dataset.subjects[i]['meta_data']
            res_image = sitk.GetImageFromArray(np.multiply(res_list[i], mask))
            res_image.CopyInformation(meta_data)
            sitk.WriteImage(res_image, res_path)
        else:
            res_path = '../tests/{}/{}_{}/res/res_{}.npy'.format(args.data, args.method, args.split, subject_name)
            np.save(res_path, np.multiply(res_list[i], mask))
        print('Model Results Saved at {}'.format(res_path))


def smooth(ves, s=1):
    if isinstance(ves, torch.Tensor):
        sm = gaussian_filter(ves.data.numpy(), sigma=s)
        smoothed = torch.Tensor(sm)
    else:
        smoothed = gaussian_filter(ves, sigma=s)
    return smoothed


def get_best_dice_threshold(response, label, thresholds):
    best_thresh, best_dice = None, -1
    n = int(len(thresholds) / 5.0)
    for thresh in thresholds[::n]:
        bin_response = (response >= thresh) + 0.0
        curr_dice = evaluate(label, bin_response, metric="DSC")
        if curr_dice > best_dice:
            best_thresh = thresh
            best_dice = curr_dice
    print("Got best dice {:.4f} at threshold {}".format(best_dice, best_thresh))
    return best_thresh


def full_seg_metrics(args):
    best_thresholds = []

    # create the trainer
    trainer = Trainer(config_file=args.config_file)
    # get dataloader
    data_loader = trainer.get_data_loader(train=(args.split == 'train'), shuffle=False)

    # result folder
    flux_folder = '../tests/{}/{}_{}/res'.format(args.data, args.method, args.split)

    # start the inference
    for subject in tqdm(data_loader.dataset.subjects, ncols=80, ascii=True):
        filename = 'res_{}.nii.gz' if len(subject['image'].shape) == 4 else 'res_{}.npy'
        response_file = os.path.join(flux_folder, filename.format(subject['subject_name']))
        if len(subject['image'].shape) == 4:
            response = sitk.GetArrayFromImage(sitk.ReadImage(response_file))
        else:
            response = np.load(response_file)
        label = subject['label'][0] > 0.5
        response = smooth(response, s=0.3)
        _, _, thresholds = metrics.roc_curve(label.reshape(-1), response.reshape(-1), pos_label=1)
        curr_best_thresh = get_best_dice_threshold(response, label, thresholds)
        best_thresholds.append(curr_best_thresh)
    final_threshold = np.mean(best_thresholds)

    # final_threshold = 0.05
    print('Final Threshold is {}.'.format(final_threshold))

    # record the metrics
    records, measures = {}, ['AUC', 'ACC', 'AHD', 'DSC', 'SENS', 'SPEC', 'PREC']
    for subject in tqdm(data_loader.dataset.subjects, ncols=80, ascii=True):
        filename = 'res_{}.nii.gz' if len(subject['image'].shape) == 4 else 'res_{}.npy'
        response_file = os.path.join(flux_folder, filename.format(subject['subject_name']))
        if len(subject['image'].shape) == 4:
            response = sitk.GetArrayFromImage(sitk.ReadImage(response_file))
        else:
            response = np.load(response_file)
        label = subject['label'][0] > 0.5
        response = smooth(response, s=0.3)
        # binarize the response
        bin_response = (response >= final_threshold)
        bin_response = remove_small_objects(bin_response, min_size=15)
        bin_response = bin_response + 0.0
        # calculate metrics
        for m in measures:
            if m not in records.keys():
                records[m] = []
            if m == 'AUC':
                # response = (response - response.min()) / (response.max() - response.min())
                records[m].append(metrics.roc_auc_score(label.flatten(), response.flatten()))
            else:
                records[m].append(evaluate(label, bin_response, metric=m))

    # show the final averaged metrics
    for k in records.keys():
        print('Mean of {}: {}'.format(k, np.mean(records[k])))


if __name__ == '__main__':
    arguments = parser.parse_args()
    # model_output(arguments)
    full_seg_metrics(arguments)
