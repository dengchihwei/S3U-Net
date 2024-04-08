# -*- coding = utf-8 -*-
# @File Name : PTF
# @Date : 2023/8/14 18:18
# @Author : zhiweideng
# @E-mail : zhiweide@usc.edu


import torch
import torch.nn.functional as F

from loss import get_sampling_vec


def construct_P(step_len, k1, k2=None):
    """
    generate propagation matrix
    :param step_len: sample length for single step, [b, 1]
    :param k1: first curvature for line-end points, 2D: [b]; 3D: [b]
    :param k2: second curvature for line-end points, 3D: [b]
    :return: P, propagation matrix, 2D: [b, 3, 3]; 3D: [b, 4, 4]
    """
    k1 = k1.clone() + torch.ones_like(k1) * 1e-20
    # 2d case
    if k2 is None:
        b = k1.shape[0]
        skt = torch.sin(k1 * step_len)
        ckt = torch.cos(k1 * step_len)
        # construct the propagation matrix
        P = torch.zeros(b, 3, 3)
        P[:, 0, 0] = torch.ones(b)
        P[:, 0, 1] = torch.div(skt, k1)
        P[:, 0, 2] = torch.div(1.0 - ckt, k1)
        P[:, 1, 1] = ckt
        P[:, 1, 2] = skt
        P[:, 2, 1] = -skt
        P[:, 2, 2] = ckt
    # 3d case
    else:
        b = k1.shape[0]
        k2 = k2.clone() + torch.ones_like(k2) * 1e-20
        k = torch.sqrt(k1 ** 2 + k2 ** 2)
        skt = torch.sin(k * step_len)
        ckt = torch.cos(k * step_len)
        # construct the propagation matrix
        P = torch.zeros(b, 4, 4)
        P[:, 0, 0] = torch.ones(b)
        P[:, 0, 1] = torch.div(skt, k)
        P[:, 0, 2] = torch.div(torch.mul(k1, (1.0 - ckt)), k ** 2)
        P[:, 0, 3] = torch.div(torch.mul(k2, (1.0 - ckt)), k ** 2)
        P[:, 1, 1] = ckt
        P[:, 1, 2] = torch.div(torch.mul(k1, skt), k)
        P[:, 1, 3] = torch.div(torch.mul(k2, skt), k)
        P[:, 2, 1] = torch.div(-torch.mul(k1, skt), k)
        P[:, 2, 2] = torch.div(k2 ** 2 + torch.mul(k1 ** 2, ckt), k ** 2)
        P[:, 2, 3] = torch.div(torch.mul(torch.mul(k1, k2), (ckt - 1.0)), k ** 2)
        P[:, 3, 1] = torch.div(-torch.mul(k2, skt), k)
        P[:, 3, 2] = torch.div(torch.mul(torch.mul(k1, k2), (ckt - 1.0)), k ** 2)
        P[:, 3, 3] = torch.div(k1 ** 2 + torch.mul(k2 ** 2, ckt), k ** 2)
    return P


def propagate_frame(frame, step_len, k1, k2=None):
    """
    generate next frame based on current frame and curvature
    :param frame: current parallel transport frame, shape of [b, DN, 3, 2], 3D: [b, DN, 4, 3]
    :param step_len: sample length for single step, float
    :param k1: first curvature learnt from the network, 2D: [b,]; 3D: [b,]
    :param k2: second curvature learnt from the network, 3D: [b,]
    :return: next frame, shape of [b, 3, 2], 3D: [b, 4, 3]
    """
    next_frames = []
    dir_num = frame.shape[1]
    for i in range(dir_num):
        # get the P matrix
        curr_P = construct_P(step_len, k1, k2).to(frame.device)
        next_frames.append(torch.bmm(curr_P, frame[:, i]))
    next_frame = torch.stack(next_frames, dim=1)
    return next_frame


def validate(k1, k2, step_len):
    import numpy as np
    sin, cos = np.sin, np.cos
    k = np.sqrt(k1 ** 2 + k2 ** 2)
    kt = k * step_len
    P = np.array([
        [1, sin(kt) / k, k1 * (1 - np.cos(kt)) / (k ** 2), k2 * (1 - cos(kt)) / (k ** 2)],
        [0, cos(kt), k1 * sin(kt) / k, k2 * sin(kt) / k],
        [0, -k1 * sin(kt) / k, (k2 ** 2 + k1 ** 2 * cos(kt)) / (k ** 2), k1 * k2 * (cos(kt) - 1) / (k ** 2)],
        [0, -k2 * sin(kt) / k, k1 * k2 * (cos(kt) - 1) / (k ** 2), (k1 * k1 + k2 * k2 * cos(kt)) / (k ** 2)]
    ])
    print(P)


'''---------- Functions for Curve Sampling ----------'''


def get_rotation_matrix_3d(src_dir, des_dir):
    """
    Compute a matrix R that rotates v1 to align with v2
    :param src_dir: 3D image [B, 3]. Optimal direction
    :param des_dir: 3-dim 1d vector
    :return: rotation_matrices, 3 by 3 rotation matrix
    """
    b, c = src_dir.size()                                                   # dimension of the vector
    src_dir = F.normalize(src_dir, dim=-1)                                  # normalize the vectors and flatten pixels
    des_dir = F.normalize(des_dir, dim=0)                                   # normalize the destination vectors
    des_dir = des_dir.repeat((b, 1))                                        # destination shape [B, 3]
    k = torch.bmm(src_dir.unsqueeze(2), des_dir.unsqueeze(1))               # [B, 3, 3]
    k -= torch.bmm(des_dir.unsqueeze(2), src_dir.unsqueeze(1))              # [B, 3, 3]
    rotation_matrices = torch.eye(c).repeat((b, 1, 1))                      # initialize the rotation matrix
    angle = torch.sum(src_dir * des_dir, dim=1)[:, None, None]              # the angle between the vectors [B,]
    rotation_matrices += k + torch.bmm(k, k) / (1 + angle)                  # construct the reconstruction matrix
    pos_indices = torch.argwhere(torch.abs(angle - 1.0) < 1.0e-10)          # find edge cases, same direction
    neg_indices = torch.argwhere(torch.abs(angle + 1.0) < 1.0e-10)          # find edge cases, opposite direction
    rotation_matrices[pos_indices] = torch.eye(c)                           # same direction,  identity
    rotation_matrices[neg_indices] = -torch.eye(c)                          # opposite direction, reverse identity
    return rotation_matrices


def get_local_frame(optimal_dir):
    """
    Compute the local frame based on the optimal dir
    :param optimal_dir: Optimal direction, 2D image [B, 2], 3D image [B, 3]
    :return: local frames, 2D image [B, H, W, 2, 2], 3D image [B, H, W, D, 3, 3]
    """
    if optimal_dir.shape[-1] == 3:
        x_axis = torch.tensor([1, 0, 0]) + 0.0
        y_axis = torch.tensor([0, 1, 0]) + 0.0
        z_axis = torch.tensor([0, 0, 1]) + 0.0
        rt_mtx = get_rotation_matrix_3d(optimal_dir, z_axis)
        new_x_axis = torch.matmul(rt_mtx, x_axis)
        new_y_axis = torch.matmul(rt_mtx, y_axis)
        frame = torch.stack((new_x_axis, new_y_axis, optimal_dir), dim=-1)
    else:
        index = torch.LongTensor([1, 0]).to(optimal_dir.device)
        ort_dir1 = torch.index_select(optimal_dir, -1, index)               # [B, 2]
        ort_dir1[:, 1] = - ort_dir1[:, 1]                                   # [B, 2]
        ort_dir1 = F.normalize(ort_dir1, dim=-1) + 1e-10                    # [B, 2]
        frame = torch.stack((optimal_dir, ort_dir1), dim=-2)                # [B, 2, 2]
    return frame


def model_tube(flux, init_frame, mean_rad, probe_num, sample_pt_num, step_len, k1, k2=None):
    """
    Model the tube structure of the vessel
    :param flux: 2D image [B, 1, H, W], 3D image [B, 1, H, W, D]. Vesselness response
    :param init_frame: initial image frame, [B, DN, 3, 2], [B, DN, 4, 3], in image space
    :param mean_rad: scalar field of mean radius, [B,]
    :param probe_num: number of probing along the direction under current curvatures
    :param sample_pt_num: number of sampling on the cross-sectional plane
    :param step_len: step length of each probe, in pixels, [B,]
    :param k1: first curvature, [B,]
    :param k2: second curvature, [B，]
    :return: response_prior
    """
    device, dim = flux.device, flux.dim()                                       # dimension of response, 2d or 3d
    b, c, h, w = flux.shape[:4]
    d = flux.shape[4] if dim == 5 else None
    dir_num = init_frame.shape[1]
    indices = torch.LongTensor((2, 1, 0) if dim == 5 else (1, 0)).to(device)    # indices to swap the axis
    prior, path = torch.zeros(b, dir_num, device=device), []                    # initialize the accumulated vesselness
    sample_pt_num_2 = sample_pt_num if dim == 5 else 1                          # sample num for K2 direction
    curr_frame = init_frame                                                     # initial image frame, normalized
    mean_rad = mean_rad.unsqueeze(-1).unsqueeze(-1).repeat(1, dir_num, 1)
    for i in range(probe_num):
        xt = torch.select(curr_frame, dim=-2, index=0)                          # image coordinates in image space
        Tt = torch.select(curr_frame, dim=-2, index=1)                          # normalized vessel direction
        K1 = torch.select(curr_frame, dim=-2, index=2)                          # normalized k1 direction
        K2 = torch.select(curr_frame, dim=-2, index=3) if dim == 5 else None    # normalized k3 direction
        path.append(xt)
        for j in torch.linspace(-1.0, 1.0, sample_pt_num):
            for k in torch.linspace(-1.0, 1.0, sample_pt_num_2):
                sample_dir = K1 * j + K2 * k if dim == 5 else K1 * j            # sample direction of cross-section
                sample_dir = F.normalize(sample_dir, dim=-1)                    # normalization
                sample_pos = xt + sample_dir * mean_rad                         # sample position over the image
                sample_pos = torch.index_select(sample_pos, -1, indices)        # swap the x and z axis
                # change to sample space
                sample_pos[..., 0] = 2.0 * sample_pos[..., 0] / h - 1
                sample_pos[..., 1] = 2.0 * sample_pos[..., 1] / w - 1
                if dim == 5:
                    sample_pos[..., 2] = 2.0 * sample_pos[..., 2] / d - 1
                    sample_pos = sample_pos.unsqueeze(dim=1)
                # sample the flux according to the positions
                sample_pos = sample_pos.unsqueeze(dim=1).float()
                curr_response = F.grid_sample(flux, sample_pos, align_corners=True)
                curr_response = curr_response.view(b, dir_num)
                prior += curr_response
        curr_frame = propagate_frame(curr_frame, step_len, k1, k2)              # propagate the image frame
    # prior /= (sample_pt_num * sample_pt_num_2 * probe_num)                    # normalization
    path = torch.stack(path, dim=-2)                                             # [B, N, PN, 2], [B, N, PN, 3]
    return prior, path


def sample_curve(flux, rads, sample_num, dir_num=64, prob_num=10, cs_pt_num=5, step_len=0.25):
    """
    Sample the curves over the whole images
    :param flux: 2D image [B, 1, H, W], 3D image [B, 1, H, W, D]. Vesselness response
    :param rads: 2D image [B, R, H, W], 3D image [B, R, H, W, D]. Estimated radius
    :param sample_num: number of curves to sample, int
    :param dir_num: number of sphere or circle sample direction, int
    :param prob_num: number of probing for every curve, int
    :param cs_pt_num: number of sampling on the cross-sectional plane, int
    :param step_len: step length of each probe, in pixels, float
    :return:
    """
    device, dim = flux.device, flux.dim()                                   # dimension of directions, 2d or 3d
    b, _, h, w = flux.shape[:4]
    d = flux.shape[4] if dim == 5 else None
    c = 3 if dim == 5 else 2
    # get the center elements
    cx, cy = h // 2, w // 2
    if dim == 5:
        cz = d // 2
        end_rads = rads[:, :, cx, cy, cz]
        coords = torch.tensor([cx, cy, cz]).repeat(b, dir_num, 1, 1).to(device)
    else:
        end_rads = rads[:, :, cx, cy]
        coords = torch.tensor([cx, cy]).repeat(b, dir_num, 1, 1).to(device)
    avg_rad = torch.mean(end_rads, dim=-1)                              # mean radius, [B,]
    min_rad, _ = torch.min(end_rads, dim=-1)                            # min radius, [B,]
    min_rad = torch.ones(min_rad.shape) * 0.5
    # create containers for optimal solutions
    optimal_prior = torch.zeros(b, device=device)                       # initial flux prior, [B,]
    optimal_k1 = torch.zeros(b, device=device)                          # optimal first curvature [B,]
    optimal_k2 = torch.zeros(b, device=device) if dim == 5 else None    # optimal second curvature [B,]
    optimal_path = torch.zeros(b, 2, prob_num, c, device=device)        # [B, prob_num, 2/3]
    # start to sample curves
    sample_dirs = get_sampling_vec(dir_num, dim).to(device)
    sample_space = torch.linspace(-0.1, 0.1, sample_num)
    all_frames = [get_local_frame(F.normalize(sample_dirs[i].repeat((b, 1)), dim=-1)) for i in range(dir_num)]
    all_frames = torch.stack(all_frames, dim=1)
    init_frame_1 = torch.cat((coords, all_frames), dim=-2)              # original direction
    init_frame_2 = torch.cat((coords, -all_frames), dim=-2)             # opposite direction
    for i in range(sample_num):
        k1 = torch.ones(b, device=device) * sample_space[i] / avg_rad / 2.0
        k2 = torch.ones(b, device=device) * sample_space[i] / avg_rad / 2.0 if dim == 5 else None
        # model the tube and calculate prior
        prior1, path1 = model_tube(flux, init_frame_1, avg_rad, prob_num, cs_pt_num, min_rad * step_len, k1, k2)
        prior2, path2 = model_tube(flux, init_frame_2, avg_rad, prob_num, cs_pt_num, min_rad * step_len, k1, k2)
        # get the best prior under current curvature sampling
        _, indices1 = torch.max(prior1, dim=1)
        prior1 = torch.stack([prior1[j, indices1[j]] for j in range(b)], dim=0)
        path1 = torch.stack([path1[j, indices1[j]] for j in range(b)], dim=0)
        _, indices2 = torch.max(prior2, dim=1)
        prior2 = torch.stack([prior2[j, indices2[j]] for j in range(b)], dim=0)
        path2 = torch.stack([path2[j, indices2[j]] for j in range(b)], dim=0)
        # find the most possible path
        response_prior = prior1 + prior2
        # indices = torch.where(prior2 > prior1)
        # final_path = path1.clone()
        # final_path[indices] = path2[indices]
        # update the optimal curvatures based on the priors
        optimal_k1 = torch.where(response_prior > optimal_prior, k1, optimal_k1)
        if k2 is not None:
            optimal_k2 = torch.where(response_prior > optimal_prior, k2, optimal_k2)
        # update the optimal path
        combined_path = torch.stack((path1, path2), dim=1)
        indices = torch.where(response_prior > optimal_prior)
        optimal_path[indices] = combined_path[indices]
        # update the optimal priors
        optimal_prior = torch.where(response_prior > optimal_prior, response_prior, optimal_prior)
    optimal_res = {
        'prior': optimal_prior,
        'k1': optimal_k1,
        'k2': optimal_k2,
        'path': optimal_path
    }
    return optimal_res


if __name__ == '__main__':
    # from matplotlib import pyplot as plt
    # X = torch.tensor([0.0, 0.0])
    # T = torch.tensor([0.0, 1.0])
    # K = torch.tensor([1.0, 0.0])
    # test_frame = torch.stack([X, T, K]).repeat(1, 1, 1)
    # _k = torch.tensor([2.0])
    # step_size = 0.1
    # _curr_frame = test_frame
    # points = []
    # for _ in range(314):
    #     points.append(_curr_frame[0, 0])
    #     _curr_frame = propagate_frame(_curr_frame, step_size, _k)
    # points = torch.stack(points)
    # plt.plot(points[:, 0], points[:, 1])
    # plt.xlim([-1, 1])
    # plt.show()

    _flux = torch.randn(4, 1, 31, 31, 31)
    _estimated_rad = torch.randn(4, 128, 31, 31, 31)
    sample_curve(_flux, _estimated_rad, 32, prob_num=10, cs_pt_num=5, step_len=0.5)
