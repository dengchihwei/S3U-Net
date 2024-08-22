# -*- coding = utf-8 -*-
# @File Name : train
# @Date : 2023/5/21 16:41
# @Author : zhiweideng
# @E-mail : zhiweide@usc.edu


import os
import json
import torch
import network
import dataset
import argparse
import torch.optim as optim

from tqdm import tqdm
from pathlib import Path
from logger import Logger
from datetime import date, datetime
from collections import OrderedDict
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
import monai.optimizers.lr_scheduler as lrs


class Trainer(object):
    def __init__(self, config_file):
        self.timestamp = str(date.today()) + datetime.now().strftime("-%H")

        # set up configer of the trainer
        assert config_file is not None
        self.configer = read_json(config_file)
        # get configer of each part of the training
        if 'loss' in self.configer.keys():
            self.loss_conf = self.configer['loss']
        self.dataset_conf = self.configer['data']
        self.trainer_conf = self.configer['trainer']

        # place holder for model related attributes
        self.supervised = None
        self.logger = None
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None

    def get_logger(self):
        """
        get the customized logger
        :return: logger, logger object
        """
        log_dir = os.path.join(self.trainer_conf['checkpoint_dir'], 'loggers', self.configer['name'])
        os.makedirs(log_dir, exist_ok=True)
        log_filename = os.path.join(log_dir, self.trainer_conf['train_type'] + '-' + self.timestamp + '.txt')
        logger = Logger(log_filename)
        return logger

    def get_data_loader(self, train=True, shuffle=True):
        # determine it is for training or validation
        if train:
            b, d = self.dataset_conf['batch_size'], True
            self.dataset_conf['args']['train'] = True
        else:
            b, shuffle, d = 1, False, False
            self.dataset_conf['args']['train'] = False
        # get the dataset and data loader
        if hasattr(dataset, self.dataset_conf['type']):
            _dataset = getattr(dataset, self.dataset_conf['type'])(**self.dataset_conf['args'])
        else:
            raise ValueError('The Dataset Name is NOT VALID')
        data_loader = DataLoader(_dataset, batch_size=b, shuffle=shuffle, num_workers=0, drop_last=d)
        return data_loader

    def get_model(self):
        """
        get the model from configer
        :return: model, torch model architecture
        """
        # define architectures based on model's type
        if hasattr(network, self.configer['arch']['type']):
            model = getattr(network, self.configer['arch']['type'])(**self.configer['arch']['args'])
        else:
            raise ValueError('Model Type Not Found.')
        # model distribution for multiple GPU devices
        if self.trainer_conf['gpu_device_num'] > 1:
            model = DataParallel(model, device_ids=list(range(self.trainer_conf['gpu_device_num'])))
        if 'supervised' in self.configer['arch'].keys():
            self.supervised = self.configer['arch']['supervised']
        model = model.cuda()
        return model

    def get_optimizer(self):
        """
        get the optimizer from configer
        :return: optimizer, torch optimizer based on model architecture
        """
        opt_type = self.configer['optimizer']['type']
        lr = self.configer['optimizer']['learning_rate']
        decay = self.configer['optimizer']['weight_decay']
        if opt_type == 'Adam':
            amsgrad = self.configer['optimizer']['amsgrad']
            optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=decay, amsgrad=amsgrad)
        elif opt_type == 'AdamW':
            amsgrad = self.configer['optimizer']['amsgrad']
            optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=decay, amsgrad=amsgrad)
        elif opt_type == 'SGD':
            momentum = self.configer['optimizer']['momentum']
            optimizer = optim.SGD(self.model.parameters(), momentum=momentum, lr=lr, weight_decay=decay)
        else:
            raise ValueError('Optimizer Type Not Found.')
        return optimizer

    def get_scheduler(self):
        """
        get the scheduler of the optimizer
        :return: scheduler, optimizer scheduler
        """
        sche_type = self.configer['lr_scheduler']['type']
        if sche_type == 'StepLR':
            step_size = self.configer['lr_scheduler']['step_size']
            gamma = self.configer['lr_scheduler']['gamma']
            scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif sche_type == 'MultiStepLR':
            milestones = self.configer['lr_scheduler']['milestones']
            gamma = self.configer['lr_scheduler']['gamma']
            scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=gamma)
        elif sche_type == "WarmupCosine":
            warmup_steps = self.configer['lr_scheduler']['warmup_steps']
            num_steps = self.configer['lr_scheduler']['num_steps']
            scheduler = lrs.WarmupCosineSchedule(self.optimizer, warmup_steps=warmup_steps, t_total=num_steps)
        else:
            raise ValueError('Scheduler Type Not Found.')
        return scheduler

    def train_epoch(self, train_loader, epoch_idx):
        """
        train one epoch for the dataset
        :param train_loader: training data loaders
        :param epoch_idx: the index of the epoch
        :return: average_losses, average losses
        """
        n_samples = 0
        batch_losses_sum = {}

        # start to training
        self.model.train()
        for idx, batch in enumerate(tqdm(train_loader, desc=str(epoch_idx), unit='b', ncols=80, ascii=True)):
            images = batch['image'].cuda()

            # losses from loss function
            if self.supervised:
                gts = batch['label'].cuda()
                _, batch_losses = self.model(images, gts)
            else:
                _, batch_losses = self.model(images, self.loss_conf)

            # first batch initialization loss
            if idx == 0:
                for key in batch_losses.keys():
                    batch_losses_sum[key] = 0.0

            # step optimizer
            if not torch.isnan(batch_losses['total_loss'].mean()):
                self.optimizer.zero_grad()
                batch_losses['total_loss'].mean().backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            else:
                raise ValueError('Loss Value Explosion!!!')

            # accumulate losses through batches
            curr_batch_len = images.size(0)
            n_samples += curr_batch_len
            for key in batch_losses.keys():
                batch_losses_sum[key] += batch_losses[key].mean().detach().item() * curr_batch_len

        # learning rate decrease
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        # get the loss dicts
        average_losses = {}
        for key in batch_losses_sum.keys():
            average_losses['avg_{}'.format(key)] = batch_losses_sum[key] / n_samples
        return average_losses

    def valid_epoch(self, epoch_idx):
        """
        evaluate one epoch for the valid dataloader
        :param epoch_idx: the index of the epoch
        :return: average_total_loss, average total loss to monitor
        """
        pass

    def train(self):
        """
        train process
        :return: None
        """
        # get model related attributes
        self.logger = self.get_logger()
        self.model = self.get_model()
        self.optimizer = self.get_optimizer()
        self.lr_scheduler = self.get_scheduler()

        # resume training
        if self.trainer_conf['resume']:
            self.resume_checkpoint()

        # freeze the network or not
        if 'linear_probe' in self.trainer_conf.keys() and self.trainer_conf['linear_probe']:
            self.model.freeze_unet()
            print('Linear probing, all encoders and decoders are frozen.')

        # clear all the content of the logger file
        self.logger.flush()
        # log the config file of current running
        self.logger.write_dict(self.configer)
        # get the train loader
        train_loader = self.get_data_loader(train=True)

        # start training
        epoch_num = self.trainer_conf['epoch_num']
        for epoch in range(1, epoch_num + 1):
            # train current epoch
            losses = self.train_epoch(train_loader, epoch)
            # separate the epoch using one line '*'
            self.logger.write_block(1)
            self.logger.write('EPOCH: {}'.format(str(epoch)))
            # log the losses to the file
            for key, value in losses.items():
                message = '{}: {}'.format(str(key), value)
                self.logger.write(message)

            # save as period
            if epoch % self.trainer_conf['save_period'] == 0:
                self.save_checkpoint(epoch, losses)
            self.logger.write_block(2)

    def save_checkpoint(self, epoch_idx, losses):
        """
        save trained model check point
        :param epoch_idx: the index of the epoch
        :param losses: losses log
        :return: None
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch_idx,
            'configer': self.configer,
            'model': (self.model.module if self.trainer_conf['gpu_device_num'] > 1 else self.model).state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_loss': losses['avg_total_loss']
        }
        chkpt_dir = os.path.join(self.trainer_conf['checkpoint_dir'], self.configer['name'], self.timestamp)
        os.makedirs(chkpt_dir, exist_ok=True)
        filename = os.path.join(chkpt_dir, '{}-{}-epoch.pth'.format(self.trainer_conf['train_type'], epoch_idx))
        self.logger.write("Saving checkpoint at: {} ...".format(filename))
        torch.save(state, filename)

    def resume_checkpoint(self):
        """
        resume the trained model check point
        :return:
        """
        self.logger.write("Loading checkpoint: {} ...".format(self.trainer_conf['resume_path']))
        checkpoint = torch.load(self.trainer_conf['resume_path'])

        # load state dicts
        multi_gpu = self.trainer_conf['gpu_device_num'] > 1
        if checkpoint['configer']['arch'] != self.configer['arch']:
            print('Attention!! Checkpoint Architecture Does Not Match to The Config File.')
            print('Fine-tuning Mode. Not continue training.')
            # load the attributes that in the current model
            model_dict = (self.model.module if multi_gpu else self.model).state_dict()
            chkpt_dict = {k: v for k, v in checkpoint['model'].items() if k in model_dict}
            model_dict.update(chkpt_dict)
            (self.model.module if multi_gpu else self.model).load_state_dict(model_dict)
        else:
            print('Continue Training Mode.')
            # load the attributes that in the current model
            (self.model.module if multi_gpu else self.model).load_state_dict(checkpoint['model'])
            # load optimizer dicts
            if checkpoint['configer']['optimizer']['type'] != self.configer['optimizer']['type']:
                raise ValueError('Checkpoint Optimizer Does Not Match to The Config File.')
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print('Optimizer resumed from before.')
        self.logger.write("Resume training from epoch {}".format(checkpoint['epoch']))


def read_json(config_file):
    """
    read the json file to config the training and dataset
    :param config_file: config file path
    :return: dictionary of config keys
    """
    config_file = Path(config_file)
    with config_file.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config_file', type=str, default='./configs/drive/unet.json')


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    args = parser.parse_args()
    trainer = Trainer(args.config_file)
    trainer.train()
