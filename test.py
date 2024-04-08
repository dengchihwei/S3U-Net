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
from loss import flux_loss
from train import read_json, Trainer


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='DRIVE')
parser.add_argument('--split', type=str, default='train')
parser.add_argument('--method', type=str, default='LCN')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--config_file', type=str, default='./configs/drive/adaptive_lc.json')
parser.add_argument('--model_path', type=str,
                    default='../trained_models/DRIVE/DRIVE_ADAPTIVE_LC/2024-03-25-01/ADAPTIVE_LC-400-epoch.pth')


def model_output(args):
    # load checkpoint
    checkpoint = torch.load(args.model_path, map_location=args.device)
    assert checkpoint['configer']['arch'] == read_json(args.config_file)['arch']

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
    flux, dirs, rads = None, None, None
    flux_list, dirs_list, rads_list = [], [], []
    px, py = data_loader.dataset.patch_sizes[:2]
    pz = data_loader.dataset.patch_sizes[2] if len(data_loader.dataset.patch_sizes) == 3 else None
    for batch in tqdm(data_loader, ncols=80, ascii=True, desc='Inference: '):
        curr_image_id = batch['image_id']
        if curr_image_id != prev_image_id:
            if prev_image_id != -1:
                flux_list.append(flux.copy())
                dirs_list.append(dirs.copy())
                rads_list.append(rads.copy())
            image_shape = data_loader.dataset.subjects[curr_image_id]['image'].shape
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

            # update the results
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
    rads_list.append(rads.copy())

    # save the results lists to files
    os.makedirs("../tests/{}/{}_{}".format(args.data, args.method, args.split), exist_ok=True)
    os.makedirs("../tests/{}/{}_{}/rads".format(args.data, args.method, args.split), exist_ok=True)
    os.makedirs("../tests/{}/{}_{}/dirs".format(args.data, args.method, args.split), exist_ok=True)
    os.makedirs("../tests/{}/{}_{}/flux".format(args.data, args.method, args.split), exist_ok=True)
    assert len(flux_list) == len(dirs_list) == len(rads_list) == len(data_loader.dataset.subjects)
    for i in range(len(flux_list)):
        mask = data_loader.dataset.subjects[i]['mask']
        subject_name = data_loader.dataset.subjects[i]['subject_name']
        rads_path = '../tests/{}/{}_{}/rads/rads_{}.npy'.format(args.data, args.method, args.split, subject_name)
        dirs_path = '../tests/{}/{}_{}/dirs/dirs_{}.npy'.format(args.data, args.method, args.split, subject_name)
        np.save(rads_path, rads_list[i])
        np.save(dirs_path, dirs_list[i])
        if len(flux_list[i].shape) == 3:
            flux_path = '../tests/{}/{}_{}/flux/flux_{}.nii.gz'.format(args.data, args.method, args.split, subject_name)
            meta_data = data_loader.dataset.subjects[i]['meta_data']
            flux_image = sitk.GetImageFromArray(np.multiply(flux_list[i], mask))
            flux_image.CopyInformation(meta_data)
            sitk.WriteImage(flux_image, flux_path)
        else:
            flux_path = '../tests/{}/{}_{}/flux/flux_{}.npy'.format(args.data, args.method, args.split, subject_name)
            np.save(flux_path, np.multiply(flux_list[i], mask))
        print('Model Results Saved at {}'.format(flux_path))


if __name__ == '__main__':
    arguments = parser.parse_args()
    model_output(arguments)
