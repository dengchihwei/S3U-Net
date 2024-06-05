# -*- coding = utf-8 -*-
# @File Name : dataset
# @Date : 2023/5/21 00:12
# @Author : zhiweideng
# @E-mail : zhiweide@usc.edu


import os
import sknw
import torch
import numpy as np
import SimpleITK as SiTk
import monai.transforms as tf

from PIL import Image
from tqdm import tqdm
from ietk import methods
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter
from skimage.measure import label as connected
from skimage.morphology import binary_erosion,  binary_closing, remove_small_objects, skeletonize, skeletonize_3d


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
                    tf.CropForeground(k_divisible=[patch_sizes[0], patch_sizes[1], patch_sizes[2]], allow_smaller=True)
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
        mask = curr_subject['mask'] > 0.001 if 'mask' in curr_subject.keys() else None
        label = curr_subject['label'] > 0.5 if 'label' in curr_subject.keys() else None
        # get image, label and mask patch
        if patch_idx is not None:
            start_coord = self.get_start_coord(image_idx, image.shape, patch_idx)
            image_patch = self.crop_image_patch(image, start_coord)
            mask_patch = self.crop_image_patch(mask, start_coord) if mask is not None else None
            label_patch = self.crop_image_patch(label, start_coord) if label is not None else None
        else:
            start_coord, image_patch, mask_patch, label_patch = None, image, mask, label
        # apply the image mask
        if mask_patch is not None and len(image_patch.shape) == 2:
            image_patch = np.multiply(image_patch, mask_patch)
        # image augmentation only for 2d images
        if self.augment and label_patch is not None and len(image_patch.shape) == 2:
            image_patch, label_patch = self.flip(image_patch, label_patch)
            image_patch, label_patch = self.rotate(image_patch, label_patch)
        # convert to torch types
        image_patch = torch.from_numpy(image_patch.copy()).unsqueeze(0)
        item = {
            'image_id': image_idx,
            'image': image_patch.float()
        }
        if start_coord is not None:
            item['start_coord'] = torch.LongTensor(start_coord)
        if label_patch is not None:
            item['label'] = torch.from_numpy(label_patch.copy()).unsqueeze(0).float()
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
            # pre-process to adjust the contrast
            image = np.array(image) * 1.0 / 255.0
            image = np.clip(methods.brighten_darken(image, 'A+B+C+X'), a_min=0.0, a_max=None)
            image = np.clip(methods.sharpen(image), a_min=0.0, a_max=None)
            # select the green channel
            image = image[..., 1]                                   # green channel
            label = np.asarray(label)
            mask = np.asarray(mask)
            # normalization of image
            image = self.normalize(image)
            mask = np.array(mask > 0.5).astype(float)
            mask = binary_erosion(mask, np.ones((5, 5)))
            subject_name = image_files[i].split('/')[-1].split('.')[0]
            # current subject
            curr_subj = {
                'subject_name': subject_name,
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
            # crop the valid sizes
            if train:
                valid_poses = np.argwhere(mask > 0)
                min_x, min_y, min_z = np.clip(np.min(valid_poses, axis=0) - 8, a_min=0, a_max=None)
                max_x, max_y, max_z = np.max(valid_poses, axis=0) + 8
                mask = mask[min_x:max_x, min_y:max_y, min_z:max_z]
                image = image[min_x:max_x, min_y:max_y, min_z:max_z]
            assert image.shape == mask.shape
            patch_dim = np.ceil((np.array(image.shape) - patch_sizes) / spacings + 1).astype(int)
            patch_num = np.prod(patch_dim)
            self.total_patch_num += patch_num
            curr_subj = {
                'subject_name': folder,
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
        for folder in tqdm(subject_folders, ncols=80, ascii=True, desc='Loading Data: '):
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
                'subject_name': folder,
                'image': - self.normalize(image, (400, 0)),
                'mask': mask,
                'label': label,
                'patch_dim': patch_dim,
                'patch_num': patch_num,
                'meta_data': SiTk.ReadImage(mask_path)
            }
            if train and subject_num % 4 != 1:
                self.subjects.append(curr_subj)
                self.total_patch_num += patch_num
            elif not train and subject_num % 4 == 1:
                self.subjects.append(curr_subj)
                self.total_patch_num += patch_num


class LineEndDataset(Dataset):
    def __init__(self, patch_size, threshold, min_size, min_len=5, augment=False):
        """
        abstract class for line end connection
        :param patch_size: size of patches to crop the features
        :param threshold: threshold to binarize the flux response
        :param min_size: minimum size of connected components to remove
        :param min_size: the minimum length of each branch, int
        :param augment: augmentation only for 2D datasets
        """
        self.subjects = []
        self.line_end_locs = []
        self.patch_size = patch_size
        self.threshold = threshold
        self.min_size = min_size
        self.min_len = min_len
        self.augment = augment

    def __len__(self):
        return len(self.line_end_locs)

    def __getitem__(self, index):
        image_idx, end_index = self.line_end_locs[index]
        end_loc = self.subjects[image_idx]['ends_locs'][end_index]
        end_dir = self.subjects[image_idx]['ends_dirs'][end_index]

        # crop the image features
        flux_patch = self.crop_feature(self.subjects[image_idx]['flux'], end_loc)
        rads_patch = self.crop_feature(self.subjects[image_idx]['rads'], end_loc)
        dirs_patch = self.crop_feature(self.subjects[image_idx]['dirs'], end_loc)

        if 'mask' in self.subjects[image_idx].keys():
            mask_patch = self.crop_feature(self.subjects[image_idx]['mask'] > 0.5, end_loc)
            flux_patch = np.multiply(flux_patch, mask_patch)
        if self.augment and len(flux_patch.shape) == 2:
            flux_patch = self.flip(flux_patch)
            flux_patch = self.rotate(flux_patch)

        # convert to torch types
        item = {
            'image_id': image_idx,
            'pred_flux': torch.from_numpy(flux_patch.copy()).float(),
            'pred_rads': torch.from_numpy(rads_patch.copy()).float(),
            'pred_dirs': torch.from_numpy(dirs_patch.copy()).float(),
            'end_loc': torch.from_numpy(end_loc.copy()).unsqueeze(0).float(),
            'end_dir': torch.from_numpy(end_dir.copy()).unsqueeze(0).float()
        }
        return item

    def crop_feature(self, feature, end_loc):
        """
        crop the features centered as the line end locations
        :param feature: image features shape of [C, H, W], [C, H, W, D]
        :param end_loc: location of line end
        :return: cropped the features [C, PS, PS], [C, PS, PS, PS]
        """
        pad_size = (self.patch_size, self.patch_size)
        pad_width = ((0, 0), pad_size, pad_size, pad_size) if feature.ndim == 4 else ((0, 0), pad_size, pad_size)
        padded_feature = np.pad(feature, pad_width)
        start_x, start_y = end_loc[:2]
        end_x, end_y = end_loc[:2] + 2 * self.patch_size + 1
        if feature.ndim == 4:
            start_z = end_loc[2]
            end_z = end_loc[2] + 2 * self.patch_size + 1
            patch_feature = padded_feature[:, start_x:end_x, start_y:end_y, start_z:end_z]
        else:
            patch_feature = padded_feature[:, start_x:end_x, start_y:end_y]
        return patch_feature

    def binarize(self, flux, mask=None):
        """
        extract the locations of line ends
        :param flux: image flux shape of [H, W], [H, W, D]
        :param mask: mask of image as shape of [H, W], [H, W, D]
        :return: binary response [H, W], [H, W, D]
        """
        if mask is not None:
            flux = np.multiply(flux, mask)
        flux = gaussian_filter(flux, sigma=0.7)
        flux = flux > self.threshold
        flux = binary_closing(flux)
        # flux = median_filter(flux, size=3)
        flux = remove_small_objects(flux, min_size=self.min_size)
        return flux

    def extract_line_ends(self, skeleton):
        """
        extract the locations of line ends
        :param skeleton: skeleton of the flux of [H, W], [H, W, D]
        :return: end_loc, locations of line end-points [K, 2], [K, 3]
        """
        # convert the skeleton to graph
        graph = sknw.build_sknw(skeleton, iso=True, ring=True, full=True)
        # filter the invalid end points
        valid_nodes_indices, valid_nodes_dirs = self.filter_nodes(skeleton, graph, self.min_len)
        # line end detection
        nodes = graph.nodes()
        ends_locs = np.array([nodes[idx]['o'] for idx in valid_nodes_indices])
        ends_locs = ends_locs.astype(np.int32)
        ends_dirs = valid_nodes_dirs
        return ends_locs, ends_dirs

    @staticmethod
    def extract_skeleton(binary_response):
        """
        extract the locations of line ends
        :param binary_response: threshold-ed flux shape of [H, W], [H, W, D]
        :return: skeleton of the flux shape of [H, W], [H, W, D]
        """
        # skeletonize the image
        skel_func = skeletonize_3d if binary_response.ndim == 3 else skeletonize
        skeleton = skel_func(binary_response)
        return skeleton

    @staticmethod
    def filter_nodes(skeleton, graph, min_len):
        """
        filter out the nodes with short branches
        :param skeleton: skeleton of the flux of [H, W], [H, W, D]
        :param graph: graph of skeleton structure, networkx graph
        :param min_len: the minimum length of each branch, int
        :return: valid_node_indices, list of index of valid nodes
        """
        skel_labels = connected(skeleton)
        valid_nodes_indices, valid_nodes_dirs = [], []
        for n_idx in graph.nodes():
            start_pos = graph.nodes()[n_idx]['o']
            curr_label = skel_labels[start_pos[0], start_pos[1]]
            if len(graph[n_idx]) == 1:
                end_idx = list(graph[n_idx])[0]
                if len(graph[n_idx][end_idx]['pts']) > min_len or np.sum(skel_labels == curr_label) < 30:
                    next_pos = graph[n_idx][end_idx]['pts'][1]
                    curr_dir = start_pos.astype(np.float64) - next_pos.astype(np.float64)
                    curr_dir = curr_dir / np.linalg.norm(curr_dir)
                    valid_nodes_indices.append(n_idx)
                    valid_nodes_dirs.append(curr_dir)
        return valid_nodes_indices, valid_nodes_dirs

    @staticmethod
    def flip(img_patch, p=0.5):
        """
        randomly flip the image patch along x and y-axis
        :param img_patch: image patch array, [C, H, W], [C, H, W, D]
        :param p: probability of applying the flip
        :return: img_patch, flipped image patch array
                 gt_patch, flipped ground truth array
        """
        # flip the image horizontally with prob p
        if np.random.uniform() < p:
            img_patch = np.flip(img_patch, axis=1)
        # flip the image vertically with prob p
        if np.random.uniform() < p:
            img_patch = np.flip(img_patch, axis=2)
        return img_patch

    @staticmethod
    def rotate(img_patch, p=0.5):
        """
        randomly rotate the image patch among {0, 90, 180, 270} degrees
        :param img_patch: image patch array [C, H, W], [C, H, W, D]
        :param p: probability of applying the rotation
        :return: img_patch, flipped image patch array
                 gt_patch, flipped ground truth array
        """
        axes = [(1, 2), (1, 3), (2, 3)]
        if np.random.uniform() < p:
            k = np.random.randint(0, 4)
            j = np.random.randint(0, 3)
            img_patch = np.rot90(img_patch, k, axes=axes[j])
        return img_patch

    @staticmethod
    def normalize(image):
        image = (image - image.min()) / (image.max() - image.min())
        return image


class DriveLineEndDataset(LineEndDataset):
    def __init__(self, data_dir, train=True, patch_size=30, threshold=0.19, min_size=30, min_len=5, augment=False):
        super(DriveLineEndDataset, self).__init__(patch_size=patch_size, threshold=threshold, min_size=min_size,
                                                  min_len=min_len, augment=augment)
        split = 'train' if train else 'valid'
        data_dir = os.path.join(data_dir, 'LCN_{}'.format(split))
        mask_files = sorted(os.listdir(os.path.join(data_dir, 'mask')))
        flux_files = sorted(os.listdir(os.path.join(data_dir, 'flux')))
        dirs_files = sorted(os.listdir(os.path.join(data_dir, 'dirs')))
        rads_files = sorted(os.listdir(os.path.join(data_dir, 'rads')))
        for i in tqdm(range(len(flux_files)), ncols=80, ascii=True, desc='Loading Data: '):
            mask = np.array(Image.open(os.path.join(data_dir, 'mask', mask_files[i])).convert('L'))
            mask = binary_erosion(mask, footprint=np.ones((8, 8)))
            flux = self.normalize(np.load(os.path.join(data_dir, 'flux', flux_files[i]))) * 10.0
            dirs = np.load(os.path.join(data_dir, 'dirs', dirs_files[i]))
            rads = np.load(os.path.join(data_dir, 'rads', rads_files[i]))
            # start post-process
            binary_response = self.binarize(flux, mask)                         # thresholding
            skeleton = self.extract_skeleton(binary_response)                   # skeletonizing
            ends_locs, ends_dirs = self.extract_line_ends(skeleton)             # extract line end-points
            for j in range(ends_locs.shape[0]):
                self.line_end_locs.append((i, j))
            curr_subject = {
                'subject_name': flux_files[i].split('.')[0],
                'flux': np.expand_dims(flux, axis=0),
                'mask': np.expand_dims(mask, axis=0),
                'dirs': dirs,
                'rads': rads,
                'skel': skeleton,
                'bin_res': binary_response,
                'ends_locs': ends_locs,
                'ends_dirs': ends_dirs
            }
            self.subjects.append(curr_subject)


class LSALineEndDataset(LineEndDataset):
    def __init__(self, data_dir, train=True, patch_size=20, threshold=0.06, min_size=30, min_len=10, augment=False):
        super(LSALineEndDataset, self).__init__(patch_size=patch_size, threshold=threshold, min_size=min_size,
                                                min_len=min_len, augment=augment)
        split = 'train' if train else 'valid'
        data_dir = os.path.join(data_dir, 'LCN_{}'.format(split))
        mask_files = sorted(os.listdir(os.path.join(data_dir, 'mask')))
        flux_files = sorted(os.listdir(os.path.join(data_dir, 'flux')))
        dirs_files = sorted(os.listdir(os.path.join(data_dir, 'dirs')))
        rads_files = sorted(os.listdir(os.path.join(data_dir, 'rads')))
        for i in tqdm(range(len(flux_files)), ncols=80, ascii=True, desc='Loading Data: '):
            mask = SiTk.GetArrayFromImage(SiTk.ReadImage(os.path.join(data_dir, 'mask', mask_files[i])))
            flux = SiTk.GetArrayFromImage(SiTk.ReadImage(os.path.join(data_dir, 'flux', flux_files[i])))
            dirs = np.load(os.path.join(data_dir, 'dirs', dirs_files[i]))
            rads = np.load(os.path.join(data_dir, 'rads', rads_files[i]))
            # start post-process
            binary_response = self.binarize(flux, mask)                     # thresholding
            skeleton = self.extract_skeleton(binary_response)               # skeletonizing
            ends_locs, ends_dirs = self.extract_line_ends(skeleton)         # extract line end-points
            for j in range(ends_locs.shape[0]):
                self.line_end_locs.append((i, j))
            curr_subject = {
                'subject_name': flux_files[i].split('.')[0],
                'meta_data': SiTk.ReadImage(os.path.join(data_dir, 'mask', mask_files[i])),
                'flux': np.expand_dims(flux, axis=0),
                'mask': np.expand_dims(mask, axis=0),
                'dirs': dirs,
                'rads': rads,
                'skel': skeleton,
                'ends_locs': ends_locs,
                'ends_dirs': ends_dirs
            }
            self.subjects.append(curr_subject)
            if i > 2:
                break


class SevenTLineEndDataset(LineEndDataset):
    def __init__(self, data_dir, train=True, patch_size=20, threshold=0.06, min_size=30, min_len=10, augment=False):
        super(SevenTLineEndDataset, self).__init__(patch_size=patch_size, threshold=threshold, min_size=min_size,
                                                   min_len=min_len, augment=augment)
        split = 'train' if train else 'valid'
        data_dir = os.path.join(data_dir, 'LCN_{}'.format(split))
        mask_files = sorted(os.listdir(os.path.join(data_dir, 'mask')))
        flux_files = sorted(os.listdir(os.path.join(data_dir, 'flux')))
        dirs_files = sorted(os.listdir(os.path.join(data_dir, 'dirs')))
        rads_files = sorted(os.listdir(os.path.join(data_dir, 'rads')))
        for i in tqdm(range(len(flux_files)), ncols=80, ascii=True, desc='Loading Data: '):
            mask = SiTk.GetArrayFromImage(SiTk.ReadImage(os.path.join(data_dir, 'mask', mask_files[i])))
            flux = SiTk.GetArrayFromImage(SiTk.ReadImage(os.path.join(data_dir, 'flux', flux_files[i])))
            dirs = np.load(os.path.join(data_dir, 'dirs', dirs_files[i]))
            rads = np.load(os.path.join(data_dir, 'rads', rads_files[i]))
            # start post-process
            binary_response = self.binarize(flux, mask)                     # thresholding
            skeleton = self.extract_skeleton(binary_response)               # skeletonizing
            ends_locs, ends_dirs = self.extract_line_ends(skeleton)         # extract line end-points
            for j in range(ends_locs.shape[0]):
                self.line_end_locs.append((i, j))
            curr_subject = {
                'subject_name': flux_files[i].split('.')[0],
                'meta_data': SiTk.ReadImage(os.path.join(data_dir, 'mask', mask_files[i])),
                'flux': np.expand_dims(flux, axis=0),
                'mask': np.expand_dims(mask, axis=0),
                'dirs': dirs,
                'rads': rads,
                'skel': skeleton,
                'ends_locs': ends_locs,
                'ends_dirs': ends_dirs
            }
            self.subjects.append(curr_subject)
            if i > 2:
                break


if __name__ == '__main__':
    # # define the datasets for unit test
    # drive_path = '/ifs/loni/faculty/shi/spectrum/zdeng/MSA_Data/VesselLearning/Datasets/DRIVE'
    # drive_train = DriveDataset(drive_path, train=True, augment=True)
    # drive_valid = DriveDataset(drive_path, train=False, augment=False)
    # print('DRIVE: Train set size: {}; Test set size: {} '.format(len(drive_train), len(drive_valid)))
    # print('DRIVE Patch Size is {}'.format(drive_train[0]['image'].shape))
    #
    # stare_path = '/ifs/loni/faculty/shi/spectrum/zdeng/MSA_Data/VesselLearning/Datasets/STARE'
    # stare_train = StareDataset(stare_path, train=True, augment=True)
    # stare_valid = StareDataset(stare_path, train=False, augment=False)
    # print('STARE: Train set size: {}; Test set size: {} '.format(len(stare_train), len(stare_valid)))
    # print('STARE Patch Size is {}'.format(stare_train[0]['image'].shape))
    #
    # hrf_path = '/ifs/loni/faculty/shi/spectrum/zdeng/MSA_Data/VesselLearning/Datasets/HRF'
    # hrf_train = HRFDataset(hrf_path, train=True, patch_sizes=[256, 256], spacings=[192, 192], augment=True)
    # hrf_valid = HRFDataset(hrf_path, train=False, patch_sizes=[256, 256], spacings=[192, 192], augment=False)
    # print('HRF: Train set size: {}; Test set size: {} '.format(len(hrf_train), len(hrf_valid)))
    # print('HRF Patch Size is {}'.format(hrf_train[0]['image'].shape))
    #
    # tubetk_path = '/ifs/loni/faculty/shi/spectrum/zdeng/MSA_Data/VesselLearning/Datasets/TubeTK'
    # tubetk_train = TubeTKDataset(tubetk_path, train=True)
    # tubetk_valid = TubeTKDataset(tubetk_path, train=False)
    # print('TubeTK: Train set size: {}; Test set size: {} '.format(len(tubetk_train), len(tubetk_valid)))
    # print('TubeTK Patch Size is {}'.format(tubetk_train[0]['image'].shape))
    #
    # vessel12_path = '/ifs/loni/faculty/shi/spectrum/zdeng/MSA_Data/VesselLearning/Datasets/vessel12'
    # vessel12_train = Vessel12Dataset(vessel12_path, train=True)
    # vessel12_valid = Vessel12Dataset(vessel12_path, train=False)
    # print('VESSEL12: Train set size: {}; Test set size: {} '.format(len(vessel12_train), len(vessel12_valid)))
    # print('VESSEL12 Patch Size is {}'.format(vessel12_train[0]['image'].shape))
    #
    # seven_t_path = '/ifs/loni/faculty/shi/spectrum/zdeng/MSA_Data/VesselLearning/Datasets/7T_Lirong/organized'
    # tof_7t_train = SevenTDataset(seven_t_path, train=True)
    # tof_7t_valid = SevenTDataset(seven_t_path, train=False)
    # print('7T: Train set size: {}; Test set size: {} '.format(len(tof_7t_train), len(tof_7t_valid)))
    # print('7T Patch Size is {}'.format(tof_7t_train[0]['image'].shape))
    #
    # lsa_path = '/ifs/loni/faculty/shi/spectrum/zdeng/MSA_Data/VesselLearning/Datasets/DarkVessels/UnilateralData'
    # lsa_train = LSADataset(lsa_path, train=True)
    # lsa_valid = LSADataset(lsa_path, train=False)
    # print('LSA: Train set size: {}; Test set size: {} '.format(len(lsa_train), len(lsa_valid)))
    # print('LSA Patch Size is {}'.format(lsa_train[0]['image'].shape))
    #
    drive_le_path = '/ifs/loni/faculty/shi/spectrum/zdeng/MSA_Data/ConnectVessel/tests/DRIVE'
    drive_le_train = DriveLineEndDataset(drive_le_path, train=True, augment=False)
    drive_le_valid = DriveLineEndDataset(drive_le_path, train=False, augment=False)
    print('DRIVE LINE END: Train set size: {}; Test set size: {} '.format(len(drive_le_train), len(drive_le_valid)))
    print('DRIVE LINE END Patch Size is {}'.format(drive_le_train[0]['end_loc'].shape))
    print('DRIVE LINE END Patch Size is {}'.format(drive_le_train[0]['pred_flux'].shape))
    print('DRIVE LINE END Patch Size is {}'.format(drive_le_train[0]['pred_dirs'].shape))
    print('DRIVE LINE END Patch Size is {}'.format(drive_le_train[0]['pred_rads'].shape))
    #
    # lsa_le_path = '/ifs/loni/faculty/shi/spectrum/zdeng/MSA_Data/ConnectVessel/tests/LSA'
    # lsa_le_train = LSALineEndDataset(lsa_le_path, train=True, augment=False)
    # lsa_le_valid = LSALineEndDataset(lsa_le_path, train=False, augment=False)
    # print('LSA LINE END: Train set size: {}; Test set size: {} '.format(len(lsa_le_train), len(lsa_le_valid)))
    # print('LSA LINE END Patch Size is {}'.format(lsa_le_valid[0]['end_loc'].shape))
    # print('LSA LINE END Patch Size is {}'.format(lsa_le_train[0]['flux'].shape))
    # print('LSA LINE END Patch Size is {}'.format(lsa_le_train[0]['dirs'].shape))
    # print('LSA LINE END Patch Size is {}'.format(lsa_le_train[0]['rads'].shape))
