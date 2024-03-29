# -*- coding = utf-8 -*-
# @File Name : postprocess
# @Date : 2023/9/18 19:33
# @Author : zhiweideng
# @E-mail : zhiweide@usc.edu


import os
import torch
import argparse
import torch.nn.functional as F

from tqdm import tqdm
from PTF import propagate_frame
from test import model_output
from loss import sample_space_to_image_space, image_space_to_sample_space


def sample_curve(response, optimal_dir, estimated_rad, sample_num, prob_num=10, pt_num=5, step_len=0.25):
    """
    Sample the curves over the whole images
    :param response: 2D image [B, 1, H, W], 3D image [B, 1, H, W, D]. Vesselness response
    :param optimal_dir: 2D image [B, H, W, 2], 3D image [B, H, W, D, 3]. Optimal direction
    :param estimated_rad: 2D image [B, H, W, N], 3D image [B, H, W, D, N]. Estimated radius
    :param sample_num: number of curves to probe
    :param prob_num: number of probing
    :param pt_num: number of sampling on the cross-sectional plane
    :param step_len: step length of each probe, in pixels
    :return:
    """
    device, dim = response.device, response.dim()                           # dimension of response, 2d or 3d
    frame = get_local_frame(optimal_dir)                                    # local frame of every pixel
    coord = sample_space_to_image_space(get_image_coord(response))          # get the image coordinates in image space
    init_frame = torch.cat((coord.unsqueeze(-2), frame), dim=-2)            # [B, H, W, 3, 2], [B, H, W, D, 4, 3]
    init_frame_r = torch.cat((coord.unsqueeze(-2), -frame), dim=-2)         # reversed frame
    mean_rad = torch.mean(estimated_rad, dim=-1, keepdim=True)              # mean radius, [B, H, W, 1], [B, H, W, D, 1]
    min_rad, _ = torch.min(estimated_rad, dim=-1, keepdim=True)             # min radius, [B, H, W, 1], [B, H, W, D, 1]
    print(min_rad.size())
    optimal_prior = torch.zeros(mean_rad.size(), device=device)             # [B, H, W, 1], [B, H, W, D, 1]
    optimal_k1 = torch.zeros(mean_rad.size(), device=device)                # [B, H, W, 1], [B, H, W, D, 1]
    optimal_k2 = torch.zeros(mean_rad.size(), device=device) if dim == 5 else None      # [B, H, W, D, 1]
    repeat_size = (1, 1, 1, 1, 6 * prob_num) if dim == 5 else (1, 1, 1, 4 * prob_num)
    optimal_path = torch.zeros(mean_rad.size(), device=device).repeat(repeat_size)      # [B, H, W, prob_num * 2]
    ranges = torch.linspace(-1.0, 1.0, sample_num)
    for i in tqdm(range(sample_num)):
        k1 = torch.ones(mean_rad.size(), device=device) * ranges[i] / mean_rad / 2.0
        k2 = torch.ones(mean_rad.size(), device=device) * ranges[i] / mean_rad / 2.0 if dim == 5 else None
        # model the tube and calculate prior
        prior_1, final_path_1 = model_tube(response, init_frame, mean_rad, prob_num, pt_num, min_rad * step_len, k1, k2)
        prior_2, final_path_2 = model_tube(response, init_frame_r, mean_rad, prob_num, pt_num, min_rad * step_len, -k1, k2)
        response_prior = prior_1 + prior_2
        # update the optimal curvatures based on the priors
        optimal_k1 = torch.where(response_prior > optimal_prior, k1, optimal_k1)
        if k2 is not None:
            optimal_k2 = torch.where(response_prior > optimal_prior, k2, optimal_k2)
        # update the optimal path
        temp_prior, temp_optimal_prior = response_prior.repeat(repeat_size), optimal_prior.repeat(repeat_size)
        final_path = torch.cat((final_path_1, final_path_2), dim=-1)
        optimal_path = torch.where(temp_prior > temp_optimal_prior, final_path, optimal_path)
        # update the optimal priors
        optimal_prior = torch.where(response_prior > optimal_prior, response_prior, optimal_prior)
    optimal_res = {
        'prior': optimal_prior,
        'k1': optimal_k1,
        'k2': optimal_k2,
        'path': optimal_path
    }
    return optimal_res


def model_tube(response, init_frame, mean_rad, probe_num, sample_pt_num, step_len, k1, k2=None):
    """
    Model the tube structure of the vessel
    :param response: 2D image [B, 1, H, W], 3D image [B, 1, H, W, D]. Vesselness response
    :param init_frame: initial image frame, [B, H, W, 3, 2], [B, H, W, D, 4, 3], in image space
    :param mean_rad: scalar field of mean radius, [B, H, W, 1], [B, H, W, D, 1]
    :param probe_num: number of probing
    :param sample_pt_num: number of sampling on the cross-sectional plane
    :param step_len: step length of each probe, in pixels
    :param k1: first curvature, [B, H, W, 1], [B, H, W, D, 1]
    :param k2: second curvature, [B, H, W, D，1]
    :return: response_prior
    """
    device = response.device
    k1 = k1.squeeze(-1)                                                     # [B, H, W], [B, H, W, D]
    k2 = k2.squeeze(-1) if k2 is not None else None                         # [B, H, W, D]
    step_len = step_len if isinstance(step_len, float) else step_len.squeeze(-1)
    indices = torch.LongTensor([2, 1, 0]) if init_frame.dim() == 6 else torch.LongTensor([1, 0])
    indices = indices.to(device)
    response_prior = torch.zeros(response.size(), device=device)            # initialize the accumulated vesselness
    curr_frame = init_frame                                                 # initial image frame, normalized
    final_path = []
    for _ in range(probe_num):
        xt = torch.select(curr_frame, dim=-2, index=0)                      # image coordinates in image space
        Tt = torch.select(curr_frame, dim=-2, index=1)                      # normalized vessel direction
        K1 = torch.select(curr_frame, dim=-2, index=2)                      # normalized k1 direction
        K2 = torch.select(curr_frame, dim=-2, index=3) if curr_frame.dim() == 6 else None
        final_path.append(xt)
        for i in torch.linspace(-1.0, 1.0, sample_pt_num):
            sample_pt_num_2 = sample_pt_num if K2 is not None else 1
            for j in torch.linspace(-1.0, 1.0, sample_pt_num_2):
                if K2 is not None:
                    sample_dir = F.normalize(K1 * i + K2 * j, dim=-1)       # [B, H, W, 2], [B, H, W, D, 3]
                else:
                    sample_dir = F.normalize(K1 * i, dim=-1)                # [B, H, W, 2], [B, H, W, D, 3]
                sample_pos = xt + sample_dir * mean_rad                     # sample position over the image
                sample_pos = torch.index_select(sample_pos, -1, indices)    # swap the x and z axis
                sample_pos = image_space_to_sample_space(sample_pos)        # change to sample space
                curr_response = F.grid_sample(response, sample_pos, align_corners=True)
                response_prior += curr_response / sample_pt_num / sample_pt_num_2 / probe_num
        curr_frame = propagate_frame(curr_frame, step_len, k1, k2)          # propagate the image frame
    response_prior = response_prior.permute(0, 2, 3, 1)                     # [B, C, H, W] --> [B, H, W, C]
    final_path = torch.flatten(torch.stack(final_path, dim=-2), start_dim=-2)
    return response_prior, final_path


def get_image_coord(image, h=None, w=None, d=None):
    """
    Generate the image coordinates from the image size
    :param image: 2D image [B, H, W, 2], 3D image [B, H, W, D, 3].
    :param h: Height of the image
    :param w: Width of the image
    :param d: Depth of the image
    :return: grid, 2D image [B, H, W, 2], 3D image [B, H, W, D, 3].
    """
    b, c = image.shape[:2]
    if h is None:
        h, w = image.shape[2:4]
        d = image.shape[4] if image.dim() == 5 else None
    dh = torch.linspace(-1.0, 1.0, h)
    dw = torch.linspace(-1.0, 1.0, w)
    if d:
        dd = torch.linspace(-1.0, 1.0, d)
        meshx, meshy, meshz = torch.meshgrid((dh, dw, dd), indexing='ij')
        grid = torch.stack((meshx, meshy, meshz), dim=3).repeat((b, 1, 1, 1, 1))    # [B, H, W, D, 3]
    else:
        meshx, meshy = torch.meshgrid((dh, dw), indexing='ij')
        grid = torch.stack((meshx, meshy), dim=2).repeat((b, 1, 1, 1))              # [B, H, W, 2]
    return grid.to(image.device)


def get_rotation_matrix_3d(src_dir, des_dir):
    """
    Compute a matrix R that rotates v1 to align with v2
    :param src_dir: 3D image [B, H, W, D, 3]. Optimal direction
    :param des_dir: 3-dim 1d vector
    :return: rotation_matrices, 3 by 3 rotation matrix
    """
    b, h, w, d, c = src_dir.size()                                          # dimension of the vector
    src_dir = F.normalize(src_dir, dim=-1).view(-1, c)                      # normalize the vectors and flatten pixels
    des_dir = F.normalize(des_dir, dim=0)                                   # normalize the destination vectors
    des_dir = des_dir.repeat((src_dir.size(0), 1))                          # destination shape [N(b*h*w*d), 3]
    k = torch.bmm(src_dir.unsqueeze(2), des_dir.unsqueeze(1))               # [N, 3, 3]
    k -= torch.bmm(des_dir.unsqueeze(2), src_dir.unsqueeze(1))              # [N, 3, 3]
    rotation_matrices = torch.eye(c).repeat((src_dir.size(0), 1, 1))        # initialize the rotation matrix
    angle = torch.sum(src_dir * des_dir, dim=1)[:, None, None]              # the angle between the vectors [B,]
    rotation_matrices += k + torch.bmm(k, k) / (1 + angle)                  # construct the reconstruction matrix
    pos_indices = torch.argwhere(torch.abs(angle - 1.0) < 1.0e-10)          # find edge cases, same direction
    neg_indices = torch.argwhere(torch.abs(angle + 1.0) < 1.0e-10)          # find edge cases, opposite direction
    rotation_matrices[pos_indices] = torch.eye(c)                           # same direction,  identity
    rotation_matrices[neg_indices] = -torch.eye(c)                          # opposite direction, reverse identity
    rotation_matrices = rotation_matrices.view(b, h, w, d, c, c)            # re-assemble the pixels
    return rotation_matrices


def get_local_frame(optimal_dir):
    """
    Compute the local frame based on the optimal dir
    :param optimal_dir: Optimal direction, 2D image [B, H, W, 2], 3D image [B, H, W, D, 3]
    :return: local frames, 2D image [B, H, W, 2, 2], 3D image [B, H, W, D, 3, 3]
    """
    if optimal_dir.dim() == 5:
        x_axis = torch.tensor([1, 0, 0]) + 0.0
        y_axis = torch.tensor([0, 1, 0]) + 0.0
        z_axis = torch.tensor([0, 0, 1]) + 0.0
        rt_mtx = get_rotation_matrix_3d(optimal_dir, z_axis)
        new_x_axis = torch.matmul(rt_mtx, x_axis)
        new_y_axis = torch.matmul(rt_mtx, y_axis)
        frame = torch.stack((new_x_axis, new_y_axis, optimal_dir), dim=-1)
    else:
        index = torch.LongTensor([1, 0]).to(optimal_dir.device)
        ort_dir1 = torch.index_select(optimal_dir, -1, index)               # [B, H, W, 2]
        ort_dir1[:, :, :, 1] = - ort_dir1[:, :, :, 1]                       # [B, H, W, 2]
        ort_dir1 = F.normalize(ort_dir1, dim=-1) + 1e-10                    # [B, H, W, 2]
        frame = torch.stack((optimal_dir, ort_dir1), dim=3)                 # [B, H, W, 2, 2]
    return frame


"""----------- Sample Curve From the Model's Output -----------"""


def post_proc_pipline(arguments, param_dict):
    # Get the three outputs from the network
    opt_dirs, est_rads, vesselness = model_output(arguments)
    assert len(opt_dirs) == len(est_rads) == len(vesselness)
    os.makedirs("PTF_Analysis", exist_ok=True)
    # Probe curvatures using PTF
    count = 0
    for item in zip(opt_dirs, est_rads, vesselness):
        optimal_dir, estimated_rad, response = item
        indices = torch.LongTensor([2, 1, 0]) if optimal_dir.dim() == 5 else torch.LongTensor([1, 0])
        optimal_dir = torch.index_select(optimal_dir, -1, indices.to(optimal_dir.device))
        print('Sampling Curves ...')
        optimal_res = sample_curve(response,
                                   optimal_dir,
                                   estimated_rad,
                                   param_dict['sample_num'],
                                   param_dict['prob_num'],
                                   param_dict['sample_pt_num'],
                                   param_dict['step_len'])
        print('Saving Results ...')
        for i in range(optimal_dir.size(0)):
            count += 1
            torch.save(optimal_dir[i].cpu().detach(), './PTF_Analysis/dir_{}.pt'.format(count))
            torch.save(estimated_rad[i].cpu().detach(), './PTF_Analysis/rad_{}.pt'.format(count))
            torch.save(response[i].cpu().detach(), './PTF_Analysis/res_{}.pt'.format(count))
            torch.save(optimal_res['prior'][i].cpu().detach(), './PTF_Analysis/prior_{}.pt'.format(count))
            torch.save(optimal_res['k1'][i].cpu().detach(), './PTF_Analysis/k1_{}.pt'.format(count))
            torch.save(optimal_res['path'][i].cpu().detach(), './PTF_Analysis/path_{}.pt'.format(count))
            if optimal_res['k2'] is not None:
                torch.save(optimal_res['k2'][i].cpu().detach(), './PTF_Analysis/k2_{}.pt'.format(count))


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--configer_file', type=str, default='./configs/drive/adaptive_lc.json')
parser.add_argument('-p', '--model_path', type=str, default='../trained_models/DRIVE_ADAPTIVE_LC/2023-09-13/' +
                                                            'ADAPTIVE_LC-400-epoch-2023-09-13.pt')
parser.add_argument('-d', '--device', type=str, default='cuda:1')
parser.add_argument('-s', '--split', type=str, default='valid')


if __name__ == '__main__':
    args = parser.parse_args()
    _param_dict = {
        'sample_num': 1000,
        'prob_num': 10,
        'sample_pt_num': 5,
        'step_len': 0.125
    }
    post_proc_pipline(args, _param_dict)
