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
from skimage import measure
from sklearn import metrics
from miseval import evaluate
from skimage.filters import frangi, sato, meijering
from skimage.morphology import skeletonize, skeletonize_3d, remove_small_objects, binary_erosion, binary_dilation
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt, maximum_filter, gaussian_filter

from loss import flux_loss
from train import read_json, Trainer
from boundary_IOU import boundary_iou, boundary_average_precision, boundary_pq


# LSA
# trained_models/LSA/LSA_ADAPTIVE_LC/2024-08-05-10 : 116 0.0793
# trained_models/LSA/LSA_UNET_100/2024-08-10-17
# trained_models/LSA/LSA_FT_100/2024-08-11-16
# trained_models/LSA/LSA_FT_80/2024-08-11-14
# trained_models/LSA/LSA_FT_60/2024-08-11-14
# trained_models/LSA/LSA_FT_40/2024-08-11-14
# trained_models/LSA/LSA_FT_20/2024-08-11-14
# trained_models/LSA/LSA_LP_100/2024-09-12-13

# SMILE trained_models/SMILE/SMILE_ADAPTIVE_LC/2024-08-22-02
# DRIVE trained_models/DRIVE/DRIVE_ADAPTIVE_LC/2024-08-08-16
# VESSEL12 trained_models/VESSEL12/VESSEL12_ADAPTIVE_LC/2024-08-08-16
# 7T trained_models/7T/7T_ADAPTIVE_LC/2024-08-08-17


parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:1')
parser.add_argument('--data', type=str, default='7T')
parser.add_argument('--split', type=str, default='test')
parser.add_argument('--method', type=str, default='FRANGI')
parser.add_argument('--epoch', type=int, default=108)
parser.add_argument('--date', type=str, default='2024-08-22-02')
parser.add_argument('--config_file', type=str, default='./configs/tmi/adaptive_lc.json')
parser.add_argument('--model_path', type=str, default='../trained_models/{}/{}/{}/{}-{}-epoch.pth')


def baseline_output(args):
    # get the data loaders
    trainer = Trainer(config_file=args.config_file)
    trainer.dataset_conf['batch_size'] = 1
    if args.data == 'SMILE' or args.data == '7T':
        trainer.dataset_conf['patch_sizes'] = [128, 128, 128]
    if 'augment' in trainer.dataset_conf['args'].keys():
        trainer.dataset_conf['args']['augment'] = False
    data_loader = trainer.get_data_loader(train=(args.split == 'train'), shuffle=False)

    # initialization
    prev_image_id = -1
    flux = None
    px, py = data_loader.dataset.patch_sizes[:2]
    pz = data_loader.dataset.patch_sizes[2] if len(data_loader.dataset.patch_sizes) == 3 else None

    # create the folders to save results
    result_folder = '../tests/{}/{}'.format(args.data, args.method)
    os.makedirs(result_folder, exist_ok=True)
    os.makedirs(os.path.join(result_folder, 'mask'), exist_ok=True)
    os.makedirs(os.path.join(result_folder, 'flux'), exist_ok=True)

    # save the outputs
    def save_output(_curr_image_id, _flux):
        subject_name = data_loader.dataset.subjects[_curr_image_id]['subject_name']
        # masking the output
        if 'mask' in data_loader.dataset.subjects[_curr_image_id].keys():
            mask = data_loader.dataset.subjects[_curr_image_id]['mask'][0]
            _flux = np.multiply(_flux, mask)
        if len(_flux.shape) == 3:
            flux_path = os.path.join(result_folder, 'flux/flux_{}.nii.gz'.format(subject_name))
            meta_data = data_loader.dataset.subjects[_curr_image_id]['meta_data']
            flux_image = sitk.GetImageFromArray(_flux)
            flux_image.CopyInformation(meta_data)
            sitk.WriteImage(flux_image, flux_path)
            # save the mask
            mask_path = os.path.join(result_folder, 'mask/mask_{}.nii.gz'.format(subject_name))
            _mask = data_loader.dataset.subjects[_curr_image_id]['mask'][0]
            mask_image = sitk.GetImageFromArray(_mask)
            mask_image.CopyInformation(meta_data)
            sitk.WriteImage(mask_image, mask_path)
        else:
            flux_path = os.path.join(result_folder, 'flux/flux_{}.npy'.format(subject_name))
            np.save(flux_path, _flux)
        # display
        print('Model Results Saved at {}'.format(flux_path))

    # set-up sigmas
    if args.method == 'FRANGI':
        sigmas = np.linspace(0.5, 12.0, 16)
    elif args.method == 'SATO':
        sigmas = np.linspace(0.5, 5.0, 8)
    else:
        sigmas = np.linspace(0.5, 3.0, 6)

    # start inference
    for batch in tqdm(data_loader, ncols=80, ascii=True, desc='Inference:'):
        curr_image_id = batch['image_id']
        if curr_image_id != prev_image_id:
            if prev_image_id != -1:
                # save the output data separately
                save_output(prev_image_id, flux)
            # reinitialize the output attributes
            image_shape = data_loader.dataset.subjects[curr_image_id]['image'].shape[1:]
            flux = np.zeros(image_shape)
            prev_image_id = curr_image_id

        # baseline calculation
        image_patch = batch['image']
        # filter out the invalid patches
        if abs(torch.mean(image_patch)) == 1.0:
            continue
        # calculation
        if args.method == 'FRANGI':
            v_func = frangi
        elif args.method == 'SATO':
            v_func = sato
        elif args.method == 'MEIJERING':
            v_func = meijering
        else:
            raise NotImplementedError
        # update the result
        b = image_patch.shape[0]
        for i in range(b):
            curr_image_patch = image_patch[i][0].numpy()

            curr_flux = v_func(curr_image_patch, sigmas=sigmas, black_ridges=False, mode='reflect')
            x, y = batch['start_coord'][i][:2]
            if pz is not None:
                z = batch['start_coord'][i][2]
                flux[x:x + px, y:y + py, z:z + pz] = np.maximum(flux[x:x + px, y:y + py, z:z + pz], curr_flux)
            else:
                flux[x:x + px, y:y + py] = np.maximum(flux[x:x + px, y:y + py], curr_flux.cpu())

    # save the last output
    save_output(prev_image_id, flux)


def model_output(args):
    configer = read_json(args.config_file)
    model_path = args.model_path.format(args.data,
                                        '{}_{}'.format(args.data, args.method),
                                        args.date,
                                        configer['trainer']['train_type'],
                                        args.epoch)

    # load checkpoint
    checkpoint = torch.load(model_path, map_location=args.device)
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
    # trainer = Trainer(config_file='./configs/advcid/adaptive_lc.json')
    trainer.dataset_conf['batch_size'] = 1
    if 'augment' in trainer.dataset_conf['args'].keys():
        trainer.dataset_conf['args']['augment'] = False
    data_loader = trainer.get_data_loader(train=(args.split == 'train'), shuffle=False)

    # initialization
    prev_image_id = -1
    flux, dirs, rads, asym = None, None, None, None
    px, py = data_loader.dataset.patch_sizes[:2]
    pz = data_loader.dataset.patch_sizes[2] if len(data_loader.dataset.patch_sizes) == 3 else None

    # create the folders to save results
    result_folder = '../tests/{}/{}-{}-{}-{}'.format(args.data, args.split, args.method, args.date, args.epoch)
    os.makedirs(result_folder, exist_ok=True)
    os.makedirs(os.path.join(result_folder, 'mask'), exist_ok=True)
    os.makedirs(os.path.join(result_folder, 'flux'), exist_ok=True)
    os.makedirs(os.path.join(result_folder, 'rads'), exist_ok=True)
    os.makedirs(os.path.join(result_folder, 'dirs'), exist_ok=True)
    os.makedirs(os.path.join(result_folder, 'asym'), exist_ok=True)

    # save the outputs
    def save_output(_curr_image_id, _flux, _rads=None, _dirs=None, _asym=None):
        subject_name = data_loader.dataset.subjects[_curr_image_id]['subject_name']
        # masking the output
        if 'mask' in data_loader.dataset.subjects[_curr_image_id].keys():
            mask = data_loader.dataset.subjects[_curr_image_id]['mask'][0]
            _flux = np.multiply(_flux, mask)
        if len(_flux.shape) == 3:
            flux_path = os.path.join(result_folder, 'flux/flux_{}.nii.gz'.format(subject_name))
            meta_data = data_loader.dataset.subjects[_curr_image_id]['meta_data']
            flux_image = sitk.GetImageFromArray(_flux)
            flux_image.CopyInformation(meta_data)
            sitk.WriteImage(flux_image, flux_path)
            # save the mask
            # mask_path = os.path.join(result_folder, 'mask/mask_{}.nii.gz'.format(subject_name))
            # _mask = data_loader.dataset.subjects[_curr_image_id]['mask'][0]
            # mask_image = sitk.GetImageFromArray(_mask)
            # mask_image.CopyInformation(meta_data)
            # sitk.WriteImage(mask_image, mask_path)
        else:
            flux_path = os.path.join(result_folder, 'flux/flux_{}.npy'.format(subject_name))
            np.save(flux_path, _flux)
        # save other attributes
        if _rads is not None:
            rads_path = os.path.join(result_folder, 'rads/rads_{}.npy'.format(subject_name))
            np.save(rads_path, _rads)
        if _dirs is not None:
            dirs_path = os.path.join(result_folder, 'dirs/dirs_{}.npy'.format(subject_name))
            np.save(dirs_path, _dirs)
        if _asym is not None:
            asym_path = os.path.join(result_folder, 'asym/asym_{}.npy'.format(subject_name))
            np.save(asym_path, _asym)
        # display
        print('Model Results Saved at {}'.format(flux_path))

    # start inference
    for batch in tqdm(data_loader, ncols=80, ascii=True, desc='Inference:'):
        curr_image_id = batch['image_id']
        if curr_image_id != prev_image_id:
            if prev_image_id != -1:
                # save the output data separately
                save_output(prev_image_id, flux, rads, dirs, asym)
            # reinitialize the output attributes
            image_shape = data_loader.dataset.subjects[curr_image_id]['image'].shape[1:]
            flux = np.zeros(image_shape)
            if args.method == 'ADAPTIVE_LC':
                # rads = np.zeros((trainer.loss_conf['flux_sample_num'],) + image_shape)
                rads = np.zeros((1, ) + image_shape)
                dirs = np.zeros((trainer.configer['arch']['args']['out_ch'],) + image_shape)
                asym = np.zeros(image_shape)
            prev_image_id = curr_image_id

        # model inference
        image_patch = batch['image'].to(args.device)
        # filter out the invalid patches
        if abs(torch.mean(image_patch)) == 1.0:
            continue

        with torch.no_grad():
            output, _ = model(image_patch)
            if args.method == 'ADAPTIVE_LC':
                curr_dirs = output['vessel']
                # curr_rads = output['rads']
                curr_rads = torch.mean(output['radius'], dim=1, keepdim=True)
                curr_flux, curr_asym = flux_loss(image_patch, output, trainer.loss_conf['flux_sample_num'])
            else:
                curr_flux = torch.sigmoid(output)

            # update the result
            b = curr_flux.shape[0]
            for i in range(b):
                x, y = batch['start_coord'][i][:2]
                if pz is not None:
                    z = batch['start_coord'][i][2]
                    if rads is not None:
                        rads[:, x:x+px, y:y+py, z:z+pz] = curr_rads[i].cpu().numpy()
                    if dirs is not None:
                        dirs[:, x:x+px, y:y+py, z:z+pz] = curr_dirs[i].cpu().numpy()
                    if asym is not None:
                        asym[x:x+px, y:y+py, z:z+pz] = curr_asym[i].cpu().numpy()
                    flux[x:x+px, y:y+py, z:z+pz] = np.maximum(flux[x:x+px, y:y+py, z:z+pz], curr_flux.cpu().numpy())
                else:
                    if rads is not None:
                        rads[:, x:x + px, y:y + py] = curr_rads[i].cpu().numpy()
                    if dirs is not None:
                        dirs[:, x:x + px, y:y + py] = curr_dirs[i].cpu().numpy()
                    if asym is not None:
                        asym[x:x + px, y:y + py] = curr_asym[i].cpu().numpy()
                    flux[x:x + px, y:y + py] = np.maximum(flux[x:x + px, y:y + py], curr_flux.cpu().numpy())

    # save the last output
    save_output(prev_image_id, flux, rads, dirs, asym)


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
    n = int(len(thresholds) / 5.0) if len(thresholds) > 5 else 1
    for thresh in thresholds[::n]:
        bin_response = (response >= thresh) + 0.0
        curr_dice = evaluate(label, bin_response, metric="DSC")
        if curr_dice > best_dice:
            best_thresh = thresh
            best_dice = curr_dice
    print("Got best dice {:.4f} at threshold {}".format(best_dice, best_thresh))
    return best_thresh


def separate_foreground(args, background_thresh=520):
    # create the trainer from config
    # trainer = Trainer(config_file=args.config_file)
    trainer = Trainer('./configs/smile/adaptive_lc.json')
    # get dataloader
    data_loader = trainer.get_data_loader(train=(args.split == 'train'), shuffle=False)

    # result folder
    flux_folder = '../tests/{}/{}-{}-{}-{}/flux'.format(args.data, args.split, args.method, args.date, args.epoch)
    # save the new flux
    result_folder = '../tests/{}/{}-{}-{}-{}'.format(args.data, args.split, args.method, args.date, args.epoch)
    os.makedirs(result_folder, exist_ok=True)
    os.makedirs(os.path.join(result_folder, 'new_flux'), exist_ok=True)

    for subject in tqdm(data_loader.dataset.subjects, ncols=80, ascii=True):
        filename = 'flux_{}.nii.gz' if len(subject['image'].shape) == 4 else 'flux_{}.npy'
        flux_file = os.path.join(flux_folder, filename.format(subject['subject_name']))
        flux_image = sitk.ReadImage(flux_file)
        flux = sitk.GetArrayFromImage(flux_image)
        image = subject['image'][0]
        mask = subject['_mask'][0] > 0.5
        # foreground masking
        fg_flux = np.multiply(flux, mask)
        # background masking
        bg_mask = np.multiply(image, ~mask) > background_thresh
        bg_flux = np.multiply(flux, bg_mask)
        new_flux = fg_flux + bg_flux

        new_flux_image = sitk.GetImageFromArray(new_flux)
        new_flux_image.CopyInformation(flux_image)
        new_flux_path = os.path.join(result_folder, 'new_flux/flux_{}.nii.gz'.format(subject['subject_name']))
        sitk.WriteImage(new_flux_image, new_flux_path)


def full_seg_metrics(args):
    # create the trainer
    trainer = Trainer(config_file=args.config_file)
    # get dataloader
    data_loader = trainer.get_data_loader(train=(args.split == 'train'), shuffle=False)
    print(len(data_loader.dataset))

    # result folder
    flux_folder = '../tests/{}/FRANGI/flux'.format(args.data)
    # flux_folder = '../tests/{}/fixed_{}'.format(args.data, args.split)
    # flux_folder = '../tests/{}/{}-{}-{}-{}/flux'.format(args.data, args.split, args.method, args.date, args.epoch)
    # rads_folder = '../tests/{}/{}-{}-{}-{}/rads'.format(args.data, args.split, args.method, args.date, args.epoch)

    # start the inference
    # best_thresholds = []
    # for subject in tqdm(data_loader.dataset.subjects, ncols=80, ascii=True):
    #     filename = 'flux_{}.nii.gz' if len(subject['image'].shape) == 4 else 'flux_{}.npy'
    #     response_file = os.path.join(flux_folder, filename.format(subject['subject_name']))
    #     if len(subject['image'].shape) == 4:
    #         response = sitk.GetArrayFromImage(sitk.ReadImage(response_file))
    #     else:
    #         response = np.load(response_file)
    #     label = subject['label'][0] > 0.5
    #     response = smooth(response, s=0.3)
    #     _, _, thresholds = metrics.roc_curve(label.reshape(-1), response.reshape(-1), pos_label=1)
    #     curr_best_thresh = get_best_dice_threshold(response, label, thresholds)
    #     best_thresholds.append(curr_best_thresh)
    #
    # final_threshold = np.mean(best_thresholds)
    # final_threshold = 0.0793
    # final_threshold = 0.0825
    final_threshold = 0.046
    print('Final Threshold is {}.'.format(final_threshold))

    # record the metrics
    records, measures = {}, ['AUC', 'ACC', 'DSC', 'SENS', 'SPEC', 'PREC', 'TP', 'FN', 'BIOU', 'BAP', 'BPQ']
    for subject in tqdm(data_loader.dataset.subjects, ncols=80, ascii=True):
        # filename = 'fixed_flux_{}.nii.gz' if len(subject['image'].shape) == 4 else 'flux_{}.npy'
        filename = 'flux_{}.nii.gz' if len(subject['image'].shape) == 4 else 'flux_{}.npy'
        response_file = os.path.join(flux_folder, filename.format(subject['subject_name']))
        if len(subject['image'].shape) == 4:
            response = sitk.GetArrayFromImage(sitk.ReadImage(response_file))
        else:
            response = np.load(response_file)

        # mask = subject['mask'][0] > 0.5
        # mask = binary_dilation(mask, footprint=np.ones((2, 2, 2)))
        # response = response * mask

        label = subject['label'][0] > 0.5
        # label = label * mask

        # masking for the TUbeTK dataset
        # mask = binary_erosion(subject['mask'][0], footprint=np.ones((5, 5, 5)))
        # response = np.multiply(response, mask)
        # radius masking for the TUbeTK dataset
        # radius_file = os.path.join(rads_folder, 'rads_{}.npy'.format(subject['subject_name']))
        # radius = np.load(radius_file)
        # rad_mask = smooth(radius, 1.0)
        # rad_mask = (rad_mask < 7.5).astype(float)
        # response = np.multiply(response, rad_mask)
        # binarize the response threshold
        # response = smooth(response, s=0.3)
        bin_response = (response >= final_threshold)
        # bin_response = remove_small_objects(bin_response, min_size=20)
        bin_response = bin_response + 0.0

        # calculate metrics
        for m in measures:
            if m not in records.keys():
                records[m] = []
            if m == 'AUC':
                response = (response - response.min()) / (response.max() - response.min())
                records[m].append(metrics.roc_auc_score(label.flatten(), response.flatten()))
            elif m == 'BIOU':
                records[m].append(boundary_iou(bin_response.astype(int), label.astype(int), dilation_iterations=3))
            elif m == 'BPQ':
                records[m].append(boundary_pq(bin_response.astype(int), label.astype(int), dilation_iterations=3))
            elif m == 'BAP':
                response = (response - response.min()) / (response.max() - response.min())
                records[m].append(boundary_average_precision(bin_response.astype(int),
                                                             response, label.astype(int), dilation_iterations=3))
            else:
                records[m].append(evaluate(label, bin_response, metric=m))
            if m == 'DSC':
                print(records[m][-1])

    # show the final averaged metrics
    print("Current Epoch: {}".format(args.epoch))
    for k in records.keys():
        print('Mean of {}: {}'.format(k, np.mean(records[k])))


def filter_radius(bin_res):
    skeleton = skeletonize_3d(bin_res)
    distance = distance_transform_edt(bin_res)

    skeleton_coords = np.argwhere(skeleton)
    foreground_coords = np.argwhere(bin_res)

    skeleton_tree = cKDTree(skeleton_coords)
    _, nearest_skeleton_indices = skeleton_tree.query(foreground_coords)

    skel_radius = distance[skeleton > 0]
    mapped_radius = np.zeros_like(bin_res, dtype=float)

    # Assign the radius values to the corresponding foreground voxels
    mapped_radius[tuple(foreground_coords.T)] = skel_radius[nearest_skeleton_indices]
    mapped_radius = maximum_filter(mapped_radius, size=9) * bin_res
    mapped_radius = (mapped_radius < 2.5) * (mapped_radius > 0.0) * 1.0
    return mapped_radius


'''----------- Topological Metrics --------------'''


def get_largest_CC(v_p):
    labels = measure.label(v_p)
    largestCC = labels == np.argmax(np.bincount(labels.flat, weights=v_p.flat))
    return largestCC


def betti_zero_error(v_p, v_l):
    bn_v = len(np.unique(measure.label(v_p))) - 1
    bn_l = len(np.unique(measure.label(v_l))) - 1
    b0_err = abs(bn_v - bn_l)
    return b0_err


def cl_score(v, s):
    return np.sum(v*s)/np.sum(s)


def clDice(v_p, v_l):
    if len(v_p.shape) == 2:
        t_prec = cl_score(v_p, skeletonize(v_l))
        t_sens = cl_score(v_l, skeletonize(v_p))
    else:
        t_prec = cl_score(v_p, skeletonize_3d(v_l))
        t_sens = cl_score(v_l, skeletonize_3d(v_p))
    return 2 * (t_prec * t_sens) / (t_prec + t_sens)


def euler_error(v_p, v_l):
    en_p = measure.euler_number(v_p)
    en_v = measure.euler_number(v_l)
    x_err = abs(en_p - en_v)
    return x_err


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
    arguments = parser.parse_args()
    baseline_output(arguments)
    # model_output(arguments)
    # separate_foreground(arguments, 520)
    # full_seg_metrics(arguments)

    # threshold = 0.0793
    # threshold = 0.0825

    # hems = [1, 2]
    # indices = [1, 5, 9, 13, 17, 21, 25]
    # indices = ['007', '017']
    # dices_ori, dices_large_ori, cl_dices_ori, b0_errors_ori, euler_errors_ori = [], [], [], [], []
    # dices_fix, dices_large_fix, cl_dices_fix, b0_errors_fix, euler_errors_fix = [], [], [], [], []
    #
    # # for hem in hems:
    # for index in indices:
    #     # image = sitk.ReadImage('/Users/zhiweideng/Desktop/NICR/DarkVessels/UnilateralData/label_h{}_subj_{}.nii'.format(hem, index))
    #     image = sitk.ReadImage('/Users/zhiweideng/Desktop/NICR/VesselAnalysis/SMILE/valid_set/validate_label/sub{}.nii'.format(index))
    #     data = sitk.GetArrayFromImage(image)
    #
    #     # res = sitk.ReadImage('/Users/zhiweideng/Desktop/LSA_POST/flux/flux_h{}_subj_{}.nii.gz'.format(hem, index))
    #     res = sitk.ReadImage('/Users/zhiweideng/Desktop/new_flux/flux_sub{}.nii.gz'.format(index))
    #     flux = sitk.GetArrayFromImage(res) > threshold
    #     flux = remove_small_objects(flux, min_size=15)
    #
    #     # res = sitk.ReadImage('/Users/zhiweideng/Desktop/LSA_POST/fixed_valid/fixed_flux_h{}_subj_{}.nii.gz'.format(hem, index))
    #     res = sitk.ReadImage('/Users/zhiweideng/Desktop/fixed_valid/fixed_flux_sub{}.nii.gz'.format(index))
    #     fixed_flux = sitk.GetArrayFromImage(res) > threshold
    #     fixed_flux = remove_small_objects(fixed_flux, min_size=15)
    #
    #     dices_ori.append(evaluate(data, flux, metric='DSC'))
    #     dices_fix.append(evaluate(data, fixed_flux, metric='DSC'))
    #
    #     dices_large_ori.append(evaluate(data, get_largest_CC(flux), metric='DSC'))
    #     dices_large_fix.append(evaluate(data, get_largest_CC(fixed_flux), metric='DSC'))
    #
    #     cl_dices_ori.append(clDice(flux, data))
    #     cl_dices_fix.append(clDice(fixed_flux, data))
    #
    #     b0_errors_ori.append(betti_zero_error(flux, data))
    #     b0_errors_fix.append(betti_zero_error(fixed_flux, data))
    #
    #     euler_errors_ori.append(euler_error(flux, data))
    #     euler_errors_fix.append(euler_error(fixed_flux, data))
    #
    # print("ORI Dice:", np.mean(dices_ori))
    # print("FIX Dice:", np.mean(dices_fix))
    # print("ORI Dice Large:", np.mean(dices_large_ori))
    # print("FIX Dice Large:", np.mean(dices_large_fix))
    # print("ORI CL Dice:", np.mean(cl_dices_ori))
    # print("FIX CL Dice:", np.mean(cl_dices_fix))
    # print("ORI BETTI 0:", np.mean(b0_errors_ori))
    # print("FIX BETTI 0:", np.mean(b0_errors_fix))
    # print("ORI EULER ERROR:", np.mean(euler_errors_ori))
    # print("FIX EULER ERROR:", np.mean(euler_errors_fix))
