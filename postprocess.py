# -*- coding = utf-8 -*-
# @File Name : postprocess
# @Date : 2023/9/18 19:33
# @Author : zhiweideng
# @E-mail : zhiweide@usc.edu


import os
import torch
import argparse
import numpy as np
import SimpleITK as sitk

from tqdm import tqdm
from loss import get_sampling_vec
from PTF import sample_curve, sample_arc
from torch.utils.data import DataLoader
from dataset import DriveLineEndDataset, LSALineEndDataset, SMILELineEndDataset
from test import smooth

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', type=str, default='LSA')
parser.add_argument('-g', '--device', type=str, default='cpu')
parser.add_argument('-s', '--split', type=str, default='valid')
parser.add_argument('-b', '--batch_size', type=int, default=2048)


drive_le_path = '../tests/DRIVE'
lsa_le_path = '../tests/LSA/'
smile_le_path = '../tests/SMILE/valid-ADAPTIVE_LC-2024-08-22-02-135'


def postprocess(args, params):
    # get the dataset as indicated
    if args.data == 'DRIVE':
        _dataset = DriveLineEndDataset(drive_le_path, train=args.split == 'train', threshold=0.19,
                                       min_len=8, min_size=30, augment=False)
    elif args.data == 'LSA':
        _dataset = LSALineEndDataset(lsa_le_path, train=args.split == 'train', threshold=0.0793,
                                     min_len=3, min_size=10, augment=False)
    else:
        raise ValueError('No such dataset type... ')
    data_loader = DataLoader(_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # copy the flux outside the dataset
    image_fluxes, subject_names = [], []
    for s in _dataset.subjects:
        image_fluxes.append(s['flux'].copy())
        subject_names.append(s['subject_name'])

    # loop over the dataset to post-process all the images
    for batch in tqdm(data_loader, ncols=80, ascii=True, desc='Post-process: '):
        flux = batch['pred_flux'].to(args.device)
        rads = batch['pred_rads'].to(args.device)
        dirs = batch['end_dir'].to(args.device)
        # sampling directions
        sampling_vecs = get_sampling_vec(params['dir_num'], dim=flux.ndim).to(args.device)
        similarities = torch.matmul(dirs, sampling_vecs.T).squeeze(1)
        _, indices = torch.max(similarities, dim=1)
        # sample the curves from
        optimal_res = sample_arc(flux, rads, **params)
        # # find the best match direction
        # _, indices = torch.max(optimal_res['prior'], dim=1)
        # # get the opposite direction
        # indices = indices - params['dir_num'] // 2
        # indices = indices + (indices < 0) * params['dir_num']
        # extract the optimal path
        batch_indices = torch.arange(flux.shape[0])
        optimal_path = optimal_res['path'][batch_indices, indices]
        optimal_prior = optimal_res['prior'][batch_indices, indices]
        # optimal_k1 = optimal_res['k1'][batch_indices, indices]
        # test
        for i in range(flux.shape[0]):
            if optimal_prior[i] < 0.03:
                continue
            curr_flux = flux[i][0].detach().cpu().numpy()                               # [H, W], [H, W, D]
            curr_idx = batch['image_id'][i].detach().cpu().numpy()
            curr_loc = batch['end_loc'][i][0].round().int().detach().cpu().numpy()      # [2,], [3,]
            curr_path = optimal_path[i].round().int().detach().cpu().numpy()            # [prob_num, 2/3]
            # filter the invalid positions
            patch_size = curr_flux.shape[0]
            curr_path = np.maximum(curr_path, 0)
            curr_path = np.minimum(curr_path, patch_size-1)
            if curr_flux.ndim == 3:
                curr_flux[curr_path[:, 0], curr_path[:, 1], curr_path[:, 2]] = 20.0
            else:
                curr_flux[curr_path[:, 0], curr_path[:, 1]] = 20.0
            # update the flux image in the dataset
            image_fluxes[curr_idx] = paste_patch(image_fluxes[curr_idx], curr_flux, curr_loc)

    # for i in range(len(image_fluxes)):
    #     curr_ends_locs = _dataset.subjects[i]['ends_locs']
    #     for j in range(curr_ends_locs.shape[0]):
    #         image_fluxes[i][:, curr_ends_locs[j, 0], curr_ends_locs[j, 1]] = 30.0

    # save the flux as image
    os.makedirs("../tests/{}/fixed_{}".format(args.data, args.split), exist_ok=True)
    for i in range(len(image_fluxes)):
        curr_flux = image_fluxes[i][0]
        if curr_flux.ndim == 3:
            flux_path = '../tests/{}/fixed_{}/fixed_{}.nii.gz'.format(args.data, args.split, subject_names[i])
            meta_data = _dataset.subjects[i]['meta_data']
            flux_image = sitk.GetImageFromArray(curr_flux)
            flux_image.CopyInformation(meta_data)
            sitk.WriteImage(flux_image, flux_path)
        else:
            flux_path = '../tests/{}/fixed_{}/fixed_{}.npy'.format(args.data, args.split, subject_names[i])
            np.save(flux_path, curr_flux)


def postprocess_new(args, params):
    # get the dataset as indicated
    if args.data == 'DRIVE':
        _dataset = DriveLineEndDataset(drive_le_path, train=args.split == 'train', threshold=0.19,
                                       min_len=8, min_size=30, augment=False)
    elif args.data == 'LSA':
        _dataset = LSALineEndDataset(lsa_le_path, train=args.split == 'train', threshold=0.0793,
                                     min_len=6, min_size=8, augment=False)
    elif args.data == 'SMILE':
        _dataset = SMILELineEndDataset(smile_le_path, patch_size=35, threshold=0.0825, min_len=6,
                                       min_size=12, augment=False)
    else:
        raise ValueError('No such dataset type... ')
    data_loader = DataLoader(_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # copy the flux outside the dataset
    image_fluxes, connections, subject_names = [], [], []
    for s in _dataset.subjects:
        image_fluxes.append(s['padded_flux'].copy())
        connections.append(np.zeros_like(s['padded_flux'].copy()))
        subject_names.append(s['subject_name'])

    # loop over the dataset to post-process all the images
    for batch in tqdm(data_loader, ncols=80, ascii=True, desc='Post-process: '):
        flux = batch['pred_flux'].to(args.device)
        rads = batch['pred_rads'].to(args.device)
        dirs = batch['end_dir'].squeeze(1).to(args.device)
        # sample the curves from
        print('start to sample')
        optimal_res = sample_curve(flux, rads, dirs, **params)
        # extract the optimal path
        optimal_path = optimal_res['path']
        optimal_prior = optimal_res['prior']
        # paste the image back to fluxes
        for i in range(flux.shape[0]):
            curr_flux = flux[i][0].detach().cpu().numpy()                               # [H, W], [H, W, D]
            curr_connections = np.zeros_like(curr_flux)
            curr_idx = batch['image_id'][i].detach().cpu().numpy()
            curr_loc = batch['end_loc'][i][0].round().int().detach().cpu().numpy()      # [2,], [3,]
            curr_path = optimal_path[i].round().int().detach().cpu().numpy()            # [prob_num, 2/3]
            curr_prior = optimal_prior[i].cpu().numpy()
            # prevent the overshoot problem
            for j in range(curr_path.shape[0], 0, -1):
                x, y, z = curr_path[j-1, 0], curr_path[j-1, 1], curr_path[j-1, 2]
                if curr_flux[x, y, z] >= 0.04:
                    curr_path = curr_path[:j]
                    curr_prior = curr_prior[:j]
                    break
            # filter out the false positive paths
            if np.mean(curr_prior) < 0.05 or len(curr_path) < 40 or np.any(curr_prior < 0.003):
                continue
            # filter the invalid positions
            patch_size = curr_flux.shape[0]
            curr_path = np.maximum(curr_path, 0)
            curr_path = np.minimum(curr_path, patch_size - 1)
            if curr_flux.ndim == 3:
                curr_flux[curr_path[:, 0], curr_path[:, 1], curr_path[:, 2]] = 20.0
                curr_connections[curr_path[:, 0], curr_path[:, 1], curr_path[:, 2]] = 20.0
            else:
                curr_flux[curr_path[:, 0], curr_path[:, 1]] = 20.0
                curr_connections[curr_path[:, 0], curr_path[:, 1]] = 20.0
            # update the flux image in the dataset
            image_fluxes[curr_idx] = paste_patch(image_fluxes[curr_idx], curr_flux, curr_loc)
            connections[curr_idx] = paste_patch(connections[curr_idx], curr_connections, curr_loc)

    # save the flux as image
    os.makedirs("../tests/{}/fixed_{}".format(args.data, args.split), exist_ok=True)
    os.makedirs("../tests/{}/connect_{}".format(args.data, args.split), exist_ok=True)
    for i in range(len(image_fluxes)):
        patch_size = _dataset.patch_size
        curr_connect = connections[i][0]
        curr_flux = image_fluxes[i][0] + smooth(curr_connect, s=0.5)
        if curr_flux.ndim == 3:
            # fixed flux image
            curr_flux = curr_flux[patch_size:-patch_size, patch_size:-patch_size, patch_size:-patch_size]
            flux_path = '../tests/{}/fixed_{}/fixed_{}.nii.gz'.format(args.data, args.split, subject_names[i])
            meta_data = _dataset.subjects[i]['meta_data']
            flux_image = sitk.GetImageFromArray(curr_flux)
            flux_image.CopyInformation(meta_data)
            sitk.WriteImage(flux_image, flux_path)
            # image of connections
            curr_connect = curr_connect[patch_size:-patch_size, patch_size:-patch_size, patch_size:-patch_size]
            connect_path = '../tests/{}/connect_{}/connect_{}.nii.gz'.format(args.data, args.split, subject_names[i])
            connect_image = sitk.GetImageFromArray(curr_connect)
            connect_image.CopyInformation(meta_data)
            sitk.WriteImage(connect_image, connect_path)
        else:
            curr_flux = curr_flux[patch_size:-patch_size, patch_size:-patch_size]
            flux_path = '../tests/{}/fixed_{}/fixed_{}.npy'.format(args.data, args.split, subject_names[i])
            np.save(flux_path, curr_flux)
            curr_connect = curr_connect[patch_size:-patch_size, patch_size:-patch_size]
            connect_path = '../tests/{}/connect_{}/connect_{}.nii.gz'.format(args.data, args.split, subject_names[i])
            np.save(connect_path, curr_connect)


def paste_patch(padded_flux, patch, line_end_loc):
    assert patch.shape[0] == patch.shape[1]
    patch_size = patch.shape[0] // 2
    # pad_size = (patch_size, patch_size)
    # pad_width = ((0, 0), pad_size, pad_size, pad_size) if flux.ndim == 4 else ((0, 0), pad_size, pad_size)
    # padded_flux = np.pad(flux, pad_width)
    start_x, start_y = line_end_loc[:2]
    end_x, end_y = line_end_loc[:2] + 2 * patch_size + 1
    if patch.ndim == 3:
        start_z = line_end_loc[2]
        end_z = line_end_loc[2] + 2 * patch_size + 1
        padded_flux[0, start_x:end_x, start_y:end_y, start_z:end_z] += (patch > 19) * 5
        # new_flux = padded_flux[:, patch_size:-patch_size, patch_size:-patch_size, patch_size:-patch_size]
    else:
        padded_flux[0, start_x:end_x, start_y:end_y] += (patch > 19) * 5
        # new_flux = padded_flux[:, patch_size:-patch_size, patch_size:-patch_size]
    # assert new_flux.shape == flux.shape
    return padded_flux


def visualize_process(args):
    # get the dataset as indicated
    if args.data == 'DRIVE':
        _dataset = DriveLineEndDataset(drive_le_path, train=args.split == 'train', threshold=0.19,
                                       min_len=8, min_size=30, augment=False)
    elif args.data == 'LSA':
        _dataset = LSALineEndDataset(lsa_le_path, train=args.split == 'train', threshold=0.0793,
                                     min_len=6, min_size=8, augment=False)
    elif args.data == 'SMILE':
        _dataset = SMILELineEndDataset(smile_le_path, threshold=0.0825, min_len=6, min_size=12, augment=False)
    else:
        raise ValueError('No such dataset type... ')

    # save all the image for visualization
    os.makedirs('../tests/{}/{}_vis'.format(args.data, args.split), exist_ok=True)

    # get the intermediate outcomes
    for subject in tqdm(_dataset.subjects, ncols=80, ascii=True, desc='visualization:'):
        skeleton = subject['skel']
        ends_locs = subject['ends_locs']
        meta_data = subject['meta_data']

        # assign a different value for line end points
        # for loc in ends_locs:
        #     x, y, z = loc
        #     skeleton[x, y, z] = 10.0

        skel_path = '../tests/{}/{}_vis/skeleton_{}.nii.gz'.format(args.data, args.split, subject['subject_name'])
        skel_image = sitk.GetImageFromArray(skeleton)
        skel_image.CopyInformation(meta_data)
        sitk.WriteImage(skel_image, skel_path)


if __name__ == '__main__':
    arguments = parser.parse_args()
    # _params = {
    #     # 'dir_num': 16,
    #     'sample_num': 16,
    #     'prob_num': 500,
    #     'cs_pt_num': 4,
    #     'max_k': 0.1,
    #     'step_len': 0.06
    # }
    # postprocess_new(arguments, _params)
    visualize_process(arguments)
