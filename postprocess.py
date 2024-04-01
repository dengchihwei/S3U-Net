# -*- coding = utf-8 -*-
# @File Name : postprocess
# @Date : 2023/9/18 19:33
# @Author : zhiweideng
# @E-mail : zhiweide@usc.edu


import os
import argparse
import numpy as np

from tqdm import tqdm
from PTF import sample_curve
from torch.utils.data import DataLoader
from dataset import DriveLineEndDataset, LSALineEndDataset

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', type=str, default='DRIVE')
parser.add_argument('-g', '--device', type=str, default='cuda:0')
parser.add_argument('-s', '--split', type=str, default='train')
parser.add_argument('-b', '--batch_size', type=int, default=512)


drive_le_path = '/ifs/loni/faculty/shi/spectrum/zdeng/MSA_Data/ConnectVessel/tests/DRIVE'
lsa_le_path = '/ifs/loni/faculty/shi/spectrum/zdeng/MSA_Data/ConnectVessel/tests/LSA'


def postprocess(args, params):
    # get the dataset as indicated
    if args.data == 'DRIVE':
        _dataset = DriveLineEndDataset(drive_le_path, train=args.split == 'train', augment=False)
    elif args.data == 'LSA':
        _dataset = LSALineEndDataset(lsa_le_path, train=args.split == 'train', augment=False)
    else:
        raise ValueError('No such dataset type... ')
    data_loader = DataLoader(_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    image_fluxes = [s['flux'].copy() for s in _dataset.subjects]
    subject_names = [s['subject_name'] for s in _dataset.subjects]

    # loop over the dataset to post-process all the images
    for batch in tqdm(data_loader, ncols=80, ascii=True, desc='Post-process: '):
        flux = batch['flux'].to(args.device)
        rads = batch['rads'].to(args.device)
        optimal_res = sample_curve(flux, rads, **params)
        for i in range(flux.shape[0]):
            curr_flux = flux[i][0].detach().cpu().numpy()                               # [H, W], [H, W, D]
            curr_loc = batch['end_loc'][i][0].int().detach().cpu().numpy()                    # [2,], [3,]
            curr_idx = batch['image_id'][i].detach().cpu().numpy()
            curr_path = optimal_res['path'][i].int().detach().cpu().numpy()             # [prob_num, 2/3]
            # filter the invalid positions
            patch_size = curr_flux.shape[0]
            curr_path = np.maximum(curr_path, 0)
            curr_path = np.minimum(curr_path, patch_size-1)
            if curr_flux.ndim == 3:
                curr_flux[curr_path[:, 0], curr_path[:, 1], curr_path[:, 2]] = 1.0
            else:
                curr_flux[curr_path[:, 0], curr_path[:, 1]] = 1.0
            # update the flux image in the dataset
            image_fluxes[curr_idx] = paste_patch(image_fluxes[curr_idx], curr_flux, curr_loc)

    # save the flux as image
    os.makedirs("../tests/{}/fixed_{}".format(args.data, args.split), exist_ok=True)
    for i in range(len(image_fluxes)):
        flux_path = '../tests/{}/fixed_{}/fixed_{}.npy'.format(args.data, args.split, subject_names[i])
        np.save(flux_path, image_fluxes[i])


def paste_patch(flux, patch, end_loc):
    assert patch.shape[0] == patch.shape[1]
    patch_size = patch.shape[0] // 2
    pad_size = (patch_size, patch_size)
    pad_width = ((0, 0), pad_size, pad_size, pad_size) if flux.ndim == 4 else ((0, 0), pad_size, pad_size)
    padded_flux = np.pad(flux, pad_width)
    start_x, start_y = end_loc[:2]
    end_x, end_y = end_loc[:2] + 2 * patch_size + 1
    if patch.ndim == 3:
        start_z = end_loc[2]
        end_z = end_loc[2] + 2 * patch_size + 1
        padded_flux[0, start_x:end_x, start_y:end_y, start_z:end_z] = patch
        new_flux = padded_flux[:, patch_size:-patch_size, patch_size:-patch_size, patch_size:-patch_size]
    else:
        padded_flux[0, start_x:end_x, start_y:end_y] = patch
        new_flux = padded_flux[:, patch_size:-patch_size, patch_size:-patch_size]
    assert new_flux.shape == flux.shape
    return new_flux


if __name__ == '__main__':
    arguments = parser.parse_args()
    _params = {
        'sample_num': 32,
        'prob_num': 60,
        'cs_pt_num': 5,
        'step_len': 0.5
    }
    postprocess(arguments, _params)
