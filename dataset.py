# -*- coding = utf-8 -*-
# @File Name : dataset
# @Date : 2023/5/21 00:12
# @Author : zhiweideng
# @E-mail : zhiweide@usc.edu


import os
import torch
import numpy as np
import SimpleITK as SiTk
import monai.transforms as tf
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from scipy.ndimage import binary_erosion


class VesselDataset(Dataset):
    def __init__(self, patch_sizes=None, spacings=None, augment=False):
        """
        abstract class for image loader
        :param patch_sizes: size of patches to crop
        :param spacings: spacings of patch cropping
        :param augment: augmentation only for 2D datasets
        """
        self.subjects = []
        self.patch_sizes = patch_sizes
        self.spacings = spacings
        self.augment = augment
        self.total_patch_num = 0
        # for 3D image dataset transform
        if patch_sizes is not None and len(patch_sizes) == 3:
            self.train_transform = tf.Compose(
                [
                    tf.CropForeground(k_divisible=[patch_sizes[0], patch_sizes[1], patch_sizes[2]], allow_smaller=True),
                    tf.ToTensor()
                ]
            )

    def __len__(self):
        return self.total_patch_num

    def __getitem__(self, index):
        # calculate the patch index is landed in which image and patch exactly
        image_idx, patch_idx = self._calc_patch_index(index) if self.patch_sizes is not None else (index, None)
        # load images
        curr_subject = self.subjects[image_idx]
        image = curr_subject['image']
        mask = curr_subject['mask'] if 'mask' in curr_subject.keys() else None
        label = curr_subject['label'] if 'label' in curr_subject.keys() else None
        # get image, label and mask patch
        if patch_idx is not None:
            start_coord = self.get_start_coord(image_idx, image.shape, patch_idx)
            image_patch = self.crop_image_patch(image, start_coord)
            mask_patch = self.crop_image_patch(mask, start_coord) if mask is not None else None
            label_patch = self.crop_image_patch(label, start_coord) if label is not None else None
        else:
            start_coord, image_patch, mask_patch, label_patch = None, image, mask, label
        # apply the image mask
        if mask_patch is not None:
            image_patch = np.multiply(image_patch, mask_patch)
        # image augmentation only for 2d images
        if self.augment and label_patch is not None and len(image_patch.shape) == 2:
            image_patch, label_patch = self.flip(image_patch, label_patch)
            image_patch, label_patch = self.rotate(image_patch, label_patch)
        # convert to torch types
        image_patch = torch.from_numpy(image_patch.copy()).unsqueeze(0)
        item = {
            'image_id': image_idx,
            'image': image_patch.float(),
            'start_coord': torch.LongTensor(start_coord) if start_coord is not None else None,
            'label': torch.from_numpy(label_patch.copy()).unsqueeze(0).float() if label_patch is not None else None
        }
        return item

    def _calc_patch_index(self, index):
        """ Convert the global index to image and patch index """
        image_idx = 0
        while index >= self.subjects[image_idx]['patch_num']:
            index -= self.subjects[image_idx]['patch_num']
            image_idx += 1
        patch_index = index
        return image_idx, patch_index

    def get_start_coord(self, image_idx, image_shape, patch_idx):
        """
        get the image patch's start pixel position
        :param image_idx: index of subject
        :param image_shape: the shape of the image, no need to pass all the images
        :param patch_idx: index of the image patch
        :return: start_coord, location of the start pixel of patch
        """
        ndim = len(image_shape)
        start_coord = np.zeros(ndim)
        patch_dim = self.subjects[image_idx]['patch_dim']
        for i in range(ndim):
            start_coord[i] = patch_idx % patch_dim[i]
            patch_idx = patch_idx // patch_dim[i]
        start_coord = (self.spacings * start_coord).astype(np.int16)
        end_coord = start_coord + self.patch_sizes
        # in case of exceed the boundaries
        for i in range(ndim):
            if end_coord[i] > image_shape[i]:
                end_coord[i] = image_shape[i]
                start_coord[i] = end_coord[i] - self.patch_sizes[i]
        return start_coord

    def crop_image_patch(self, image, start_coord):
        """
        get the image patch based on the patch index
        :param image: image numpy array
        :param start_coord: location of the start pixel of patch
        :return: img_patch, image patch array
        """
        h, w = start_coord[:2]
        d = start_coord[2] if len(image.shape) == 3 else None
        if d is not None:
            img_patch = image[h:h + self.patch_sizes[0], w:w + self.patch_sizes[1], d:d + self.patch_sizes[2]]
        else:
            img_patch = image[h:h + self.patch_sizes[0], w:w + self.patch_sizes[1]]
        return img_patch

    def retinal_image_read(self, image_sizes, image_files, mask_files, label_files):
        for i in tqdm(range(len(image_files)), ncols=80, ascii=True):
            image = Image.open(image_files[i])
            label = Image.open(label_files[i])
            mask = Image.open(mask_files[i]).convert('L')
            # resize the image or not
            if image_sizes is not None:
                image = image.resize(image_sizes)
                label = label.resize(image_sizes)
                mask = mask.resize(image_sizes)
            image = np.asarray(image)[..., 1]           # green channel
            label = np.asarray(label)
            mask = np.asarray(mask)
            # normalization of image
            image = self.normalize(image)
            mask = np.array(mask > 0.5).astype(float)
            mask = binary_erosion(mask, np.ones((5, 5)))
            # current subject
            curr_subj = {
                'image': - image,
                'label': label,
                'mask': mask
            }
            if self.patch_sizes is not None:
                assert self.spacings is not None
                patch_dim = np.ceil((np.array(image.shape) - self.patch_sizes) / self.spacings + 1).astype(int)
                patch_num = np.prod(patch_dim)
                curr_subj['patch_dim'] = patch_dim
                curr_subj['patch_num'] = patch_num
                self.total_patch_num += patch_num - 1   # minus one for the outer space adding up 1
            self.subjects.append(curr_subj)
            self.total_patch_num += 1

    @staticmethod
    def normalize(image, thresh=None):
        """ Min-max normalization """
        if thresh is not None:
            max_val, min_val = min(thresh[0], image.max()), max(thresh[1], image.min())
            image = np.clip(image, a_max=max_val, a_min=min_val)
        max_val, min_val = image.max(), image.min()
        return 2.0 * (image - min_val) / (max_val - min_val) - 1.0

    @staticmethod
    def flip(img_patch, gt_patch, p=0.5):
        """
        randomly flip the image patch along x and y-axis
        :param img_patch: image patch array
        :param gt_patch: ground truth array
        :param p: probability of applying the flip
        :return: img_patch, flipped image patch array
                 gt_patch, flipped ground truth array
        """
        # flip the image horizontally with prob p
        if np.random.uniform() < p:
            img_patch = np.flip(img_patch, axis=0)
            gt_patch = np.flip(gt_patch, axis=0)
        # flip the image vertically with prob p
        if np.random.uniform() < p:
            img_patch = np.flip(img_patch, axis=1)
            gt_patch = np.flip(gt_patch, axis=1)
        return img_patch, gt_patch

    @staticmethod
    def rotate(img_patch, gt_patch, p=0.5):
        """
        randomly rotate the image patch among {0, 90, 180, 270} degrees
        :param img_patch: image patch array
        :param gt_patch: ground truth array
        :param p: probability of applying the rotation
        :return: img_patch, flipped image patch array
                 gt_patch, flipped ground truth array
        """
        if np.random.uniform() < p:
            k = np.random.randint(0, 4)
            img_patch = np.rot90(img_patch, k, axes=(0, 1))
            gt_patch = np.rot90(gt_patch, k, axes=(0, 1))
        return img_patch, gt_patch

    @staticmethod
    def add_gaussian_noise(img_patch, p=0.5):
        """
        add gaussian noises to the image patch array
        :param img_patch: image patch array
        :param p: probability of adding the noises
        :return: img_patch, noised added patch
        """
        if np.random.uniform() < p:
            gaussian_noise = np.random.normal(0.01, 0.02, img_patch.shape)
            img_patch = gaussian_noise + img_patch
        return img_patch

    @staticmethod
    def read_csv(csv_file):
        """
        read cvs label files of VESSEL12 as a dictionary
        :param csv_file: .csv label file
        :return: label, label location dictionary
        """
        labels = {}
        lines = open(csv_file, "r").readlines()
        for line in lines:
            strs = line.split(',')
            x, y, z = int(strs[0]), int(strs[1]), int(strs[2])
            label = int(strs[3][0])
            labels[(z, y, x)] = label
        return labels


class DriveDataset(VesselDataset):
    def __init__(self, data_dir, image_sizes=None, train=True, patch_sizes=None, spacings=None, augment=True):
        super(DriveDataset, self).__init__(patch_sizes, spacings, augment)
        # get the data directories
        split = 'train' if train else 'test'
        mask_path = os.path.join(data_dir, split, 'mask')
        image_path = os.path.join(data_dir, split, 'images')
        label_path = os.path.join(data_dir, split, '1st_manual')
        # get the image file list
        self.mask_files = [os.path.join(mask_path, file) for file in sorted(os.listdir(mask_path))]
        self.image_files = [os.path.join(image_path, file) for file in sorted(os.listdir(image_path))]
        self.label_files = [os.path.join(label_path, file) for file in sorted(os.listdir(label_path))]
        # read 2d retina images
        self.retinal_image_read(image_sizes, self.image_files, self.mask_files, self.label_files)


class StareDataset(VesselDataset):
    def __init__(self, data_dir, image_sizes=None, train=True, patch_sizes=None, spacings=None, augment=True):
        super(StareDataset, self).__init__(patch_sizes, spacings, augment)
        # get the data directories
        mask_path = os.path.join(data_dir, 'mask')
        image_path = os.path.join(data_dir, 'images')
        label_path = os.path.join(data_dir, 'labels-ah')
        # get the image file list
        self.mask_files = [os.path.join(mask_path, file) for file in sorted(os.listdir(mask_path))]
        self.image_files = [os.path.join(image_path, file) for file in sorted(os.listdir(image_path))]
        self.label_files = [os.path.join(label_path, file) for file in sorted(os.listdir(label_path))]
        # split based on the train or test dataset
        split_index = int(0.8 * len(self.mask_files))
        self.mask_files = self.mask_files[:split_index] if train else self.mask_files[split_index:]
        self.image_files = self.image_files[:split_index] if train else self.image_files[split_index:]
        self.label_files = self.label_files[:split_index] if train else self.label_files[split_index:]
        # read 2d retina images
        self.retinal_image_read(image_sizes, self.image_files, self.mask_files, self.label_files)


class HRFDataset(VesselDataset):
    def __init__(self, data_dir, image_sizes=None, train=True, patch_sizes=None, spacings=None, augment=True):
        super(HRFDataset, self).__init__(patch_sizes, spacings, augment)
        # get the data directories
        mask_path = os.path.join(data_dir, 'mask')
        image_path = os.path.join(data_dir, 'images')
        label_path = os.path.join(data_dir, 'manual1')
        # get the image file list
        self.mask_files = [os.path.join(mask_path, file) for file in sorted(os.listdir(mask_path))]
        self.image_files = [os.path.join(image_path, file) for file in sorted(os.listdir(image_path))]
        self.label_files = [os.path.join(label_path, file) for file in sorted(os.listdir(label_path))]
        # split based on the train or test dataset
        split_index = int(0.8 * len(self.mask_files))
        self.mask_files = self.mask_files[:split_index] if train else self.mask_files[split_index:]
        self.image_files = self.image_files[:split_index] if train else self.image_files[split_index:]
        self.label_files = self.label_files[:split_index] if train else self.label_files[split_index:]
        # read 2d retina images
        self.retinal_image_read(image_sizes, self.image_files, self.mask_files, self.label_files)


class TubeTKDataset(VesselDataset):
    def __init__(self, data_dir, train=True, patch_sizes=(64, 64, 64), spacings=(48, 48, 48)):
        super(TubeTKDataset, self).__init__(patch_sizes, spacings, False)
        # get all the subject folder path
        subject_folders = sorted(os.listdir(data_dir))
        # iterate all the folders to load images
        for folder in tqdm(subject_folders, ncols=80, ascii=True):
            subject_path = os.path.join(data_dir, folder)
            modalities = os.listdir(subject_path)
            mra_path = os.path.join(subject_path, 'MRA')
            mra_path = os.path.join(mra_path, os.listdir(mra_path)[0])
            # read image
            image = SiTk.GetArrayFromImage(SiTk.ReadImage(mra_path))
            patch_dim = np.ceil((np.array(image.shape) - self.patch_sizes) / self.spacings + 1).astype(int)
            patch_num = np.prod(patch_dim)
            curr_subj = {
                'image': self.normalize(image),
                'patch_dim': patch_dim,
                'patch_num': patch_num
            }
            if 'AuxillaryData' in modalities and not train:
                label_path = os.path.join(subject_path, '{}_LABEL.npy'.format(folder))
                mask_path = os.path.join(subject_path, '{}_MASK.mha'.format(folder))
                curr_subj['label'] = np.load(label_path)
                curr_subj['mask'] = SiTk.GetArrayFromImage(SiTk.ReadImage(mask_path))
                self.subjects.append(curr_subj)
                self.total_patch_num += patch_num
            if 'AuxillaryData' not in modalities and train:
                self.subjects.append(curr_subj)
                self.total_patch_num += patch_num


class Vessel12Dataset(VesselDataset):
    def __init__(self, data_dir, train=True, patch_sizes=(64, 64, 64), spacings=(48, 48, 48)):
        super(Vessel12Dataset, self).__init__(patch_sizes, spacings, False)
        split = 'train' if train else 'test'
        # get all the subject folder path
        data_dir = os.path.join(data_dir, split)
        subject_folders = sorted(os.listdir(data_dir))
        # iterate all the folders to load images
        for folder in tqdm(subject_folders, ncols=80, ascii=True):
            subject_path = os.path.join(data_dir, folder)
            ct_path = os.path.join(subject_path, '{}.mhd'.format(folder))
            mask_path = os.path.join(subject_path, 'mask_{}.mhd'.format(folder))
            label_path = os.path.join(subject_path, '{}_Annotations.csv'.format(folder))
            # read image
            image = SiTk.GetArrayFromImage(SiTk.ReadImage(ct_path))
            mask = SiTk.GetArrayFromImage(SiTk.ReadImage(mask_path))
            patch_dim = np.ceil((np.array(image.shape) - self.patch_sizes) / self.spacings + 1).astype(int)
            patch_num = np.prod(patch_dim)
            self.total_patch_num += patch_num
            curr_subj = {
                'mask': mask,
                'image': self.normalize(image, (250, -900)),
                'patch_dim': patch_dim,
                'patch_num': patch_num
            }
            if label_path in os.listdir(subject_path):
                curr_subj['label'] = self.read_csv(label_path)
            self.subjects.append(curr_subj)


class SevenTDataset(VesselDataset):
    def __init__(self, data_dir, train=True, patch_sizes=(64, 64, 64), spacings=(48, 48, 48)):
        super(SevenTDataset, self).__init__(patch_sizes, spacings, False)
        split = 'train' if train else 'test'
        # get all the subject folder path
        data_dir = os.path.join(data_dir, split)
        subject_folders = sorted(os.listdir(data_dir))
        for folder in tqdm(subject_folders, ncols=80, ascii=True):
            subject_path = os.path.join(data_dir, folder)
            if not os.path.isdir(subject_path) or folder in ['M007', 'M008', 'mask_files']:
                continue
            mra_path = os.path.join(subject_path, '{}_TOF.nii.gz'.format(folder))
            mask_path = os.path.join(subject_path, '{}_TOF_MASKED.nii.gz'.format(folder))
            image = SiTk.GetArrayFromImage(SiTk.ReadImage(mra_path))
            mask = SiTk.GetArrayFromImage(SiTk.ReadImage(mask_path))
            patch_dim = np.ceil((np.array(image.shape) - patch_sizes) / spacings + 1).astype(int)
            patch_num = np.prod(patch_dim)
            self.total_patch_num += patch_num
            curr_subj = {
                'mask': mask,
                'image': self.normalize(image),
                'patch_dim': patch_dim,
                'patch_num': patch_num
            }
            self.subjects.append(curr_subj)


class LSADataset(VesselDataset):
    def __init__(self, data_dir, train=True, patch_sizes=(24, 24, 24), spacings=(16, 16, 16)):
        super(LSADataset, self).__init__(patch_sizes, spacings, False)
        # get all the subject folder path
        subject_folders = sorted(os.listdir(data_dir))
        for folder in tqdm(subject_folders, ncols=80, ascii=True):
            subject_path = os.path.join(data_dir, folder)
            subject_num = int(folder.split('_')[-1])
            image_path = os.path.join(subject_path, 'T1SPC_NLM03_{}.nii'.format(folder))
            mask_path = os.path.join(subject_path, 'mask_{}.nii'.format(folder))
            label_path = os.path.join(subject_path, 'label_{}.nii'.format(folder))
            # read image
            image = SiTk.GetArrayFromImage(SiTk.ReadImage(image_path))
            mask = SiTk.GetArrayFromImage(SiTk.ReadImage(mask_path))
            label = SiTk.GetArrayFromImage(SiTk.ReadImage(label_path))
            patch_dim = np.ceil((np.array(image.shape) - self.patch_sizes) / self.spacings + 1).astype(int)
            patch_num = np.prod(patch_dim)
            curr_subj = {
                'image': - self.normalize(image, (400, 0)),
                'mask': mask,
                'label': label,
                'patch_dim': patch_dim,
                'patch_num': patch_num
            }
            if train and subject_num % 4 != 1:
                self.subjects.append(curr_subj)
                self.total_patch_num += patch_num
            elif not train and subject_num % 4 == 1:
                self.subjects.append(curr_subj)
                self.total_patch_num += patch_num


if __name__ == '__main__':
    # define the datasets for unit test
    drive_path = '/ifs/loni/faculty/shi/spectrum/zdeng/MSA_Data/VesselLearning/Datasets/DRIVE'
    drive_train = DriveDataset(drive_path, train=True, augment=True)
    drive_valid = DriveDataset(drive_path, train=False, augment=False)
    print('DRIVE: Train set size: {}; Test set size: {} '.format(len(drive_train), len(drive_valid)))
    print('DRIVE Patch Size is {}'.format(drive_train[0]['image'].shape))

    stare_path = '/ifs/loni/faculty/shi/spectrum/zdeng/MSA_Data/VesselLearning/Datasets/STARE'
    stare_train = StareDataset(stare_path, train=True, augment=True)
    stare_valid = StareDataset(stare_path, train=False, augment=False)
    print('STARE: Train set size: {}; Test set size: {} '.format(len(stare_train), len(stare_valid)))
    print('STARE Patch Size is {}'.format(stare_train[0]['image'].shape))

    hrf_path = '/ifs/loni/faculty/shi/spectrum/zdeng/MSA_Data/VesselLearning/Datasets/HRF'
    hrf_train = HRFDataset(hrf_path, train=True, patch_sizes=[256, 256], spacings=[192, 192], augment=True)
    hrf_valid = HRFDataset(hrf_path, train=False, patch_sizes=[256, 256], spacings=[192, 192], augment=False)
    print('HRF: Train set size: {}; Test set size: {} '.format(len(hrf_train), len(hrf_valid)))
    print('HRF Patch Size is {}'.format(hrf_train[0]['image'].shape))

    tubetk_path = '/ifs/loni/faculty/shi/spectrum/zdeng/MSA_Data/VesselLearning/Datasets/TubeTK'
    tubetk_train = TubeTKDataset(tubetk_path, train=True)
    tubetk_valid = TubeTKDataset(tubetk_path, train=False)
    print('TubeTK: Train set size: {}; Test set size: {} '.format(len(tubetk_train), len(tubetk_valid)))
    print('TubeTK Patch Size is {}'.format(tubetk_train[0]['image'].shape))

    vessel12_path = '/ifs/loni/faculty/shi/spectrum/zdeng/MSA_Data/VesselLearning/Datasets/vessel12'
    vessel12_train = Vessel12Dataset(vessel12_path, train=True)
    vessel12_valid = Vessel12Dataset(vessel12_path, train=False)
    print('VESSEL12: Train set size: {}; Test set size: {} '.format(len(vessel12_train), len(vessel12_valid)))
    print('VESSEL12 Patch Size is {}'.format(vessel12_train[0]['image'].shape))

    seven_t_path = '/ifs/loni/faculty/shi/spectrum/zdeng/MSA_Data/VesselLearning/Datasets/7T_Lirong/organized'
    tof_7t_train = SevenTDataset(seven_t_path, train=True)
    tof_7t_valid = SevenTDataset(seven_t_path, train=False)
    print('7T: Train set size: {}; Test set size: {} '.format(len(tof_7t_train), len(tof_7t_valid)))
    print('7T Patch Size is {}'.format(tof_7t_train[0]['image'].shape))

    lsa_path = '/ifs/loni/faculty/shi/spectrum/zdeng/MSA_Data/VesselLearning/Datasets/DarkVessels/UnilateralData'
    lsa_train = LSADataset(lsa_path, train=True)
    lsa_valid = LSADataset(lsa_path, train=False)
    print('LSA: Train set size: {}; Test set size: {} '.format(len(lsa_train), len(lsa_valid)))
    print('LSA Patch Size is {}'.format(lsa_train[0]['image'].shape))
