# -*- coding = utf-8 -*-
# @File Name : PTF
# @Date : 2023/8/14 18:18
# @Author : zhiweideng
# @E-mail : zhiweide@usc.edu


import torch
import torch.nn.functional as F
import numpy as np
from skimage.morphology import binary_erosion, binary_dilation, binary_closing, remove_small_objects, skeletonize, medial_axis, thin


def construct_P(step_len, k1, k2=None):
    """
    generate propagation matrix
    :param step_len: sample length for single step, float
    :param k1: first curvature for line-end points, 2D: [b, n]; 3D: [b, n]
    :param k2: second curvature for line-end points, 3D: [b, n]
    :return: P, propagation matrix, 2D: [b, n, 3, 3]; 3D: [b, n, 4, 4]
    """
    k1 = k1.clone() + torch.ones_like(k1) * 1e-20
    # 2d case
    if k2 is None:
        b, n = k1.size()
        skt = torch.sin(k1 * step_len)
        ckt = torch.cos(k1 * step_len)
        # construct the propagation matrix
        P = torch.zeros(b, n, 3, 3)
        P[:, :, 0, 0] = torch.ones(b, n)
        P[:, :, 0, 1] = torch.div(skt, k1)
        P[:, :, 0, 2] = torch.div(1.0 - ckt, k1)
        P[:, :, 1, 1] = ckt
        P[:, :, 1, 2] = skt
        P[:, :, 2, 1] = -skt
        P[:, :, 2, 2] = ckt
    # 3d case
    else:
        b, n = k1.size()
        k2 = k2.clone() + torch.ones_like(k2) * 1e-20
        k = torch.sqrt(k1 ** 2 + k2 ** 2)
        skt = torch.sin(torch.mul(k, step_len))
        ckt = torch.cos(torch.mul(k, step_len))
        # construct the propagation matrix
        P = torch.zeros(b, n, 4, 4)
        P[:, :, 0, 0] = torch.ones(b, n)
        P[:, :, 0, 1] = torch.div(skt, k)
        P[:, :, 0, 2] = torch.div(torch.mul(k1, (1.0 - ckt)), k ** 2)
        P[:, :, 0, 3] = torch.div(torch.mul(k2, (1.0 - ckt)), k ** 2)
        P[:, :, 1, 1] = ckt
        P[:, :, 1, 2] = torch.div(torch.mul(k1, skt), k)
        P[:, :, 1, 3] = torch.div(torch.mul(k2, skt), k)
        P[:, :, 2, 1] = torch.div(-torch.mul(k1, skt), k)
        P[:, :, 2, 2] = torch.div(k2 ** 2 + torch.mul(k1 ** 2, ckt), k ** 2)
        P[:, :, 2, 3] = torch.div(torch.mul(torch.mul(k1, k2), (ckt - 1.0)), k ** 2)
        P[:, :, 3, 1] = torch.div(-torch.mul(k2, skt), k)
        P[:, :, 3, 2] = torch.div(torch.mul(torch.mul(k1, k2), (ckt - 1.0)), k ** 2)
        P[:, :, 3, 3] = torch.div(k1 ** 2 + torch.mul(k2 ** 2, ckt), k ** 2)
    return P


def propagate_frame(frame, step_len, k1, k2=None):
    """
    generate next frame based on current frame and curvature
    :param frame: current parallel transport frame, shape of [b, n, 3, 2], 3D: [b, n, 4, 3]
    :param step_len: sample length for single step, float
    :param k1: first curvature learnt from the network, 2D: [b, n]; 3D: [b, n]
    :param k2: second curvature learnt from the network, 3D: [b, n]
    :return: next frame, shape of [b, n, 3, 2], 3D: [b, n, 4, 3]
    """
    # get the P matrix
    P = construct_P(step_len, k1, k2).to(frame.device)
    # reshape the matrix, 2d case
    original_size = frame.size()
    if k2 is None:
        P = P.view(-1, 3, 3)
        frame = frame.view(-1, 3, 2)
    # reshape the matrix, 3d case
    else:
        P = P.view(-1, 4, 4)
        frame = frame.view(-1, 4, 3)
    next_frame = torch.bmm(P, frame)
    # recover the frame shape
    next_frame = next_frame.view(original_size)
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


def binarize(flux, mask, threshold, fp_size=5):
    """
    binarize the flux to get the final segmentation
    :param flux: flux response of image patch, [H, W], [H, W, D]
    :param mask: image mask [H, W], [H, W, D]
    :param threshold: threshold, float
    :param fp_size: footprint size to erode the mask, int
    :return:
    """
    footprint = np.ones((fp_size, fp_size, fp_size)) if flux.dim() == 5 else np.ones((fp_size, fp_size))
    mask_erode = binary_erosion(mask, footprint=footprint)
    flux = np.multiply(flux, mask_erode)


def crop_patches(flux, threshold, patch_size):
    flux_np = flux.detach().cpu().numpy()

    return torch.randint(0, 128, 1)


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


def extract_end_features(features, end_locs):
    """
    select features from feature maps of image original size
    :param features: feature map of image size, [B, H, W, C], [B, H, W, D, C]
    :param end_locs: locations of line end points, [B, N, 2], [B, N, 3]
    :return: selected features [B, N, C]
    """
    b, dim = features.shape[0], features.dim()
    if dim == 5:
        features = [features[i, end_locs[i, :, 0], end_locs[i, :, 1], end_locs[i, :, 2]] for i in range(b)]
    else:
        features = [features[i, end_locs[i, :, 0], end_locs[i, :, 1]] for i in range(b)]
    features = torch.stack(features, dim=0)                                 # [B, N, C]
    return features


def model_tube(flux, init_frame, mean_rad, probe_num, sample_pt_num, step_len, k1, k2=None):
    """
    Model the tube structure of the vessel
    :param flux: 2D image [B, 1, H, W], 3D image [B, 1, H, W, D]. Vesselness response
    :param init_frame: initial image frame, [B, N, 3, 2], [B, N, 4, 3], in image space
    :param mean_rad: scalar field of mean radius, [B, N, 1]
    :param probe_num: number of probing along the direction under current curvatures
    :param sample_pt_num: number of sampling on the cross-sectional plane
    :param step_len: step length of each probe, in pixels, [B, N, 1]
    :param k1: first curvature, [B, N, 1]
    :param k2: second curvature, [B, N, 1]
    :return: response_prior
    """
    device, dim = flux.device, flux.dim()                                       # dimension of response, 2d or 3d
    b, c, h, w = flux.shape[:4]
    d = flux.shape[4] if dim == 5 else None
    k1 = k1.squeeze(-1)                                                         # [B, N]
    k2 = k2.squeeze(-1) if k2 is not None else None                             # [B, N]
    indices = torch.LongTensor((2, 1, 0) if dim == 5 else (1, 0)).to(device)    # indices to swap the axis
    flux_prior = torch.zeros(flux.size(), device=device)                        # initialize the accumulated vesselness
    curr_frame, final_path = init_frame, []                                     # initial image frame, normalized
    sample_pt_num_2 = sample_pt_num if dim == 5 else 1                          # sample num for K2 direction
    for i in range(probe_num):
        xt = torch.select(curr_frame, dim=-2, index=0)                          # image coordinates in image space
        Tt = torch.select(curr_frame, dim=-2, index=1)                          # normalized vessel direction
        K1 = torch.select(curr_frame, dim=-2, index=2)                          # normalized k1 direction
        K2 = torch.select(curr_frame, dim=-2, index=3) if dim == 5 else None    # normalized k3 direction
        final_path.append(xt)
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
                # sample the flux according to the positions
                curr_response = F.grid_sample(flux, sample_pos, align_corners=True)
                flux_prior += curr_response
        curr_frame = propagate_frame(curr_frame, step_len, k1, k2)              # propagate the image frame
    flux_prior /= (sample_pt_num * sample_pt_num_2 * probe_num)                 # normalization
    response_prior = flux_prior.permute(0, 2, 3, 1)                             # [B, C, H, W] --> [B, H, W, C]
    final_path = torch.stack(final_path, dim=-2)                                # [B, N, PN, 2], [B, N, PN, 3]
    return response_prior, final_path


def sample_curve(flux, optimal_dir, estimated_rad, threshold, sample_num, prob_num=10, cs_pt_num=5, step_len=0.25):
    """
    Sample the curves over the whole images
    :param flux: 2D image [B, 1, H, W], 3D image [B, 1, H, W, D]. Vesselness response
    :param optimal_dir: 2D image [B, H, W, 2], 3D image [B, H, W, D 3]. Optimal direction
    :param estimated_rad: 2D image [B, H, W, R], 3D image [B, H, W, D, R]. Estimated radius
    :param threshold: threshold to binarize the image, float between 0 and 1
    :param sample_num: number of curves to sample, int
    :param prob_num: number of probing for every curve, int
    :param cs_pt_num: number of sampling on the cross-sectional plane, int
    :param step_len: step length of each probe, in pixels, float
    :return:
    """
    device, dim = flux.device, flux.dim()                                   # dimension of response, 2d or 3d
    b, c, h, w = flux.shape[:4]
    d = flux.shape[4] if dim == 5 else None
    end_locs = line_end_locations(flux, threshold)                          # line-end locations, [B, N, 2], [B, N, 3]
    end_dirs = line_end_features(optimal_dir, end_locs)                     # vessel directions [B, N, 2], [B, N, 3]
    end_rads = line_end_features(estimated_rad, end_locs)                   # line-end radius [B, N, R], [B, N, R]
    local_frame = get_local_frame(end_locs)                                 # initialize the PTF frame for propagation
    init_frame_1 = torch.cat((end_locs.unsqueeze(-2), local_frame), dim=-2)
    init_frame_2 = torch.cat((end_locs.unsqueeze(-2), -local_frame), dim=-2)
    avg_rad = torch.mean(estimated_rad, dim=-1, keepdim=True)               # mean radius, [B, H, W, 1], [B, H, W, D, 1]
    min_rad, _ = torch.min(estimated_rad, dim=-1, keepdim=True)             # min radius, [B, H, W, 1], [B, H, W, D, 1]
    optimal_flux_prior = torch.zeros(flux.size(), device=device)            # [B, H, W, 1], [B, H, W, D, 1]



    device, dim = flux.device, flux.dim()                           # dimension of response, 2d or 3d
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


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    X = torch.tensor([0.0, 0.0])
    T = torch.tensor([0.0, 1.0])
    K = torch.tensor([1.0, 0.0])
    test_frame = torch.stack([X, T, K]).repeat(1, 1, 1, 1)
    _k = torch.tensor([0.0]).repeat(1, 1)
    step_size = 0.1
    _curr_frame = test_frame
    points = []
    for _ in range(314):
        points.append(_curr_frame[0, 0, 0])
        _curr_frame = propagate_frame(_curr_frame, step_size, _k)
    points = torch.stack(points)
    plt.plot(points[:, 0], points[:, 1])
    plt.xlim([-1, 1])
    plt.show()
