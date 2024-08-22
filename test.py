# -*- coding = utf-8 -*-
# @File Name : test
# @Date : 2023/6/22 23:29
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

from loss import flux_loss
from train import read_json, Trainer


# LSA trained_models/LSA/LSA_ADAPTIVE_LC/2024-08-05-10
# SMILE trained_models/SMILE/SMILE_ADAPTIVE_LC/2024-08-08-17/


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='SMILE')
parser.add_argument('--split', type=str, default='valid')
parser.add_argument('--method', type=str, default='LCN')
parser.add_argument('--epoch', type=str, default='50')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--config_file', type=str, default='./configs/smile/adaptive_lc.json')
parser.add_argument('--model_path', type=str,
                    default='../trained_models/SMILE/SMILE_ADAPTIVE_LC/2024-08-08-17/ADAPTIVE_LC-{}-epoch.pth')


def model_output(args):
    configer = read_json(args.config_file)
    # load checkpoint
    checkpoint = torch.load(args.model_path.format(args.epoch), map_location=args.device)
    try:
        assert checkpoint['configer']['arch']['type'] == configer['arch']['type']
        assert checkpoint['configer']['arch']['args'] == configer['arch']['args']
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

    # initialization
    prev_image_id = -1
    flux, dirs, rads = None, None, None
    flux_list, dirs_list, rads_list = [], [], []
    px, py = data_loader.dataset.patch_sizes[:2]
    pz = data_loader.dataset.patch_sizes[2] if len(data_loader.dataset.patch_sizes) == 3 else None

    # start inference
    for batch in tqdm(data_loader, ncols=80, ascii=True, desc='Inference: '):
        curr_image_id = batch['image_id']
        if curr_image_id != prev_image_id:
            if prev_image_id != -1:
                flux_list.append(flux.copy())
                dirs_list.append(dirs.copy())
                rads_list.append(rads.mean(axis=0).copy())
            image_shape = data_loader.dataset.subjects[curr_image_id]['image'].shape[1:]

            dirs = np.zeros((trainer.configer['arch']['args']['out_ch'],) + image_shape)
            rads = np.zeros((trainer.loss_conf['flux_sample_num'],) + image_shape)
            flux = np.zeros(image_shape)
            prev_image_id = curr_image_id

        # model inference
        image_patch = batch['image'].to(args.device)
        with torch.no_grad():
            output, _ = model(image_patch)
            curr_dirs = output['vessel']
            curr_rads = output['radius']
            curr_flux = flux_loss(image_patch, output, trainer.loss_conf['flux_sample_num'])

            # update the result
            b = curr_dirs.shape[0]
            for i in range(b):
                x, y = batch['start_coord'][i][:2]
                if pz is not None:
                    z = batch['start_coord'][i][2]
                    rads[:, x:x+px, y:y+py, z:z+pz] = curr_rads[i].cpu().numpy()
                    dirs[:, x:x+px, y:y+py, z:z+pz] = curr_dirs[i].cpu().numpy()
                    flux[x:x+px, y:y+py, z:z+pz] = np.maximum(flux[x:x+px, y:y+py, z:z+pz], curr_flux.cpu().numpy())
                else:
                    rads[:, x:x + px, y:y + py] = curr_rads[i].cpu().numpy()
                    dirs[:, x:x + px, y:y + py] = curr_dirs[i].cpu().numpy()
                    flux[x:x + px, y:y + py] = np.maximum(flux[x:x + px, y:y + py], curr_flux.cpu().numpy())
    flux_list.append(flux.copy())
    dirs_list.append(dirs.copy())
    rads_list.append(rads.mean(axis=0).copy())

    # create the folders to save results
    result_folder = '../tests/{}/{}_{}_{}'.format(args.data, args.method, args.epoch, args.split)
    os.makedirs(result_folder, exist_ok=True)
    os.makedirs(os.path.join(result_folder, 'rads'), exist_ok=True)
    os.makedirs(os.path.join(result_folder, 'dirs'), exist_ok=True)
    os.makedirs(os.path.join(result_folder, 'flux'), exist_ok=True)
    assert len(flux_list) == len(dirs_list) == len(rads_list) == len(data_loader.dataset.subjects)
    # start the saving the file
    for i in range(len(flux_list)):
        subject_name = data_loader.dataset.subjects[i]['subject_name']
        rads_path = os.path.join(result_folder, 'rads/rads_{}.npy'.format(subject_name))
        dirs_path = os.path.join(result_folder, 'dirs/dirs_{}.npy'.format(subject_name))
        np.save(rads_path, rads_list[i])
        np.save(dirs_path, dirs_list[i])
        flux_data = flux_list[i]
        # masking the output
        if 'mask' in data_loader.dataset.subjects[i].keys():
            mask = data_loader.dataset.subjects[i]['mask'][0]
            flux_data = np.multiply(flux_list[i], mask)
        if len(flux_list[i].shape) == 3:
            flux_path = os.path.join(result_folder, 'flux/flux_{}.nii.gz'.format(subject_name))
            meta_data = data_loader.dataset.subjects[i]['meta_data']
            flux_image = sitk.GetImageFromArray(flux_data)
            flux_image.CopyInformation(meta_data)
            sitk.WriteImage(flux_image, flux_path)
        else:
            flux_path = os.path.join(result_folder, 'flux/flux_{}.npy'.format(subject_name))
            np.save(flux_path, flux_data)
        print('Model Results Saved at {}'.format(flux_path))


'''----------- Segmentation Metrics --------------'''


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
    flux_folder = '../tests/{}/{}_{}_{}/flux'.format(args.data, args.method, args.epoch, args.split)

    # start the inference
    for subject in tqdm(data_loader.dataset.subjects, ncols=80, ascii=True):
        filename = 'flux_{}.nii.gz' if len(subject['image'].shape) == 4 else 'flux_{}.npy'
        response_file = os.path.join(flux_folder, filename.format(subject['subject_name']))
        if len(subject['image'].shape) == 4:
            response = sitk.GetArrayFromImage(sitk.ReadImage(response_file))
        else:
            response = np.load(response_file)
        label = subject['label'][0] > 0.5
        # response = smooth(response, s=0.1)
        _, _, thresholds = metrics.roc_curve(label.reshape(-1), response.reshape(-1), pos_label=1)
        curr_best_thresh = get_best_dice_threshold(response, label, thresholds)
        best_thresholds.append(curr_best_thresh)

    final_threshold = np.mean(best_thresholds)
    print('Final Threshold is {}.'.format(final_threshold))

    # record the metrics
    records, measures = {}, ['AUC', 'ACC', 'AHD', 'DSC', 'SENS', 'SPEC', 'PREC']
    for subject in tqdm(data_loader.dataset.subjects, ncols=80, ascii=True):
        filename = 'flux_{}.nii.gz' if len(subject['image'].shape) == 4 else 'flux_{}.npy'
        response_file = os.path.join(flux_folder, filename.format(subject['subject_name']))
        if len(subject['image'].shape) == 4:
            response = sitk.GetArrayFromImage(sitk.ReadImage(response_file))
        else:
            response = np.load(response_file)
        label = subject['label'][0] > 0.5
        # response = smooth(response, s=0.1)
        # binarize the response
        bin_response = (response >= final_threshold) + 0.0
        # calculate metrics
        for m in measures:
            if m not in records.keys():
                records[m] = []
            records[m].append(evaluate(label, bin_response, metric=m))

    # show the final averaged metrics
    for k in records.keys():
        print('Mean of {}: {}'.format(k, np.mean(records[k])))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    arguments = parser.parse_args()
    # model_output(arguments)
    full_seg_metrics(arguments)
