# -*- coding = utf-8 -*-
# @File Name : test
# @Date : 2023/6/22 23:29
# @Author : zhiweideng
# @E-mail : zhiweide@usc.edu


import torch
import dataset
import network
import argparse
from tqdm import tqdm
from train import read_json
from loss import flux_loss, preproc_output
from torch.utils.data import DataLoader


def load_model(arguments):
    config = read_json(arguments.configer_file)
    checkpoint = torch.load(arguments.model_path, map_location=arguments.device)
    model = getattr(network, config['model']['type'])(**config['model']['args'])
    model.load_state_dict(checkpoint['model'])
    model.to(arguments.device).eval()
    print('load model done')
    return model


@torch.no_grad()
def model_output(arguments):
    config = read_json(arguments.configer_file)                                 # read the configer file
    data_set = getattr(dataset, config[arguments.split]['type'])(**config[arguments.split]['args'])
    data_loader = DataLoader(data_set, batch_size=12, shuffle=False)             # define data loaders
    model, device = load_model(arguments), arguments.device                     # define model and device
    optimal_dirs, estimated_rads, responses = [], [], []                        # network output list
    # start inference
    count = 0
    for batch in tqdm(data_loader, desc=str(0), unit='b'):
        images = batch['image'].to(device)
        output = model(images)
        curr_res, _ = flux_loss(images, output, config['loss']['config']['flux_sample_num'])
        curr_dir, curr_rad = preproc_output(output)
        optimal_dirs.append(curr_dir)
        estimated_rads.append(curr_rad)
        responses.append(curr_res.unsqueeze(1))
        count += 1
        if count == 1:
            break
    return optimal_dirs, estimated_rads, responses


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--configer_file', type=str, default='./configs/drive/adaptive_lc.json')
parser.add_argument('-p', '--model_path', type=str, default='../trained_models/DRIVE_ADAPTIVE_LC/2023-09-13/' +
                                                            'ADAPTIVE_LC-300-epoch-2023-09-13.pt')
parser.add_argument('-d', '--device', type=str, default='cuda:1')
parser.add_argument('-s', '--split', type=str, default='valid')


if __name__ == '__main__':
    args = parser.parse_args()
    optimal_dir_list, _, _ = model_output(args)
    for i in range(10):
        for x in range(33, 39):
            for y in range(1, 127):
                print(x, y, optimal_dir_list[0][i, x, y])
