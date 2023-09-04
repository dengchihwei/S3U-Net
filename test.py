# -*- coding = utf-8 -*-
# @File Name : test
# @Date : 2023/6/22 23:29
# @Author : zhiweideng
# @E-mail : zhiweide@usc.edu


import os
import torch
import dataset
import network
import skimage
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from train import read_json
from loss import flux_loss_asymmetry, calc_local_contrast
from torch.utils.data import DataLoader


def load_model(config, model_path, device='cpu'):
    model = getattr(network, config['model']['type'])(**config['model']['args'])
    # send to gpu devices
    model = model.to(device)
    checkpoint = torch.load(model_path, map_location=next(model.parameters()).device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model


def ours_response_2d(config, model_path, split='train', device='cpu'):
    # define the dataloader and model
    config[split]['args']['augment'] = False
    curr_loader = DataLoader(getattr(dataset, config[split]['type'])(**config[split]['args']), batch_size=1)
    model = load_model(config, model_path, device)
    # start to calculate the response
    all_responses, prev_id = [], 0
    ps = config[split]['args']['patch_size']
    response = np.zeros(np.array(config[split]['args']['img_size']))
    for item in tqdm(curr_loader, desc=str(0), unit='b'):
        image_id, image = item['image_id'], item['image'].to(device)
        x, y = item['start_coord'][0]
        if image_id != prev_id:
            all_responses.append(response.copy())
            response = np.zeros(response.shape)
            prev_id = image_id
        with torch.no_grad():
            output = model(image)
            curr_res, _ = flux_loss_asymmetry(image,
                                              output,
                                              config['loss']['config']['flux_sample_num'],
                                              config['loss']['config']['grad_dims'])
            # curr_res = calc_local_contrast(image, output['radius'], config['loss']['config']['flux_sample_num'], 10)
            patch_res = curr_res[0].cpu().detach().numpy()
            response[x:x+ps, y:y+ps] = np.maximum(patch_res, response[x:x+ps, y:y+ps])
    all_responses.append(response)
    return all_responses


def baseline_response_2d(config, method, split='train'):
    vesselness_func = getattr(skimage.filters, method)
    baseline_args = {'sigmas': np.linspace(0, 16, 32), 'black_ridges': True}
    # define the dataset
    curr_dataset = getattr(dataset, config[split]['type'])(**config[split]['args'])
    all_responses = []
    # baseline method response
    for i in range(len(curr_dataset.images)):
        image = curr_dataset.images[i]
        response = vesselness_func(image, **baseline_args)
        all_responses.append(response)
    return all_responses


def make_responses_2d(configer_file, method, model_path=None, split='train', device='cpu'):
    # get whole configurations
    config = read_json(configer_file)
    if method == 'ours':
        assert model_path is not None
        print('Start our inference...')
        all_responses = ours_response_2d(config, model_path, split, device)
    else:
        print('Start baseline enhancement...')
        all_responses = baseline_response_2d(config, method, split)
    # save the results image
    print('Start saving results...')
    os.makedirs('../tests/{}/{}'.format(method, split), exist_ok=True)
    for i in range(len(all_responses)):
        response = all_responses[i]
        np.save('../tests/{}/{}/{}.npy'.format(method, split, i), response)
        response = (response - response.min()) / (response.max() - response.min())
        image = Image.fromarray(np.uint8(response * 255.0), 'L')
        image.save('../tests/{}/{}/{}.jpg'.format(method, split, i))
    print('Inference Done.')


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config_file', type=str, default='./configs/drive/adaptive_lc.json')
parser.add_argument('-s', '--split', type=str, default='valid')
parser.add_argument('-d', '--device', type=str, default='cuda:2')
parser.add_argument('-p', '--model_path', type=str, default='../trained_models/ADAPTIVE_LC/2023-06-22/' +
                                                            'ADAPTIVE_LC-300-epoch-2023-06-22.pt')
parser.add_argument('-m', '--method', type=str, default='ours')


if __name__ == '__main__':
    args = parser.parse_args()
    make_responses_2d(args.config_file, args.method, args.model_path, args.split, args.device)
