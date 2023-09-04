# -*- coding = utf-8 -*-
# @File Name : PTF
# @Date : 2023/8/14 18:18
# @Author : zhiweideng
# @E-mail : zhiweide@usc.edu


import torch


def construct_P(step_len, k1, k2=None):
    """
    generate propagation matrix
    :param step_len: sample length for single step, int
    :param k1: first curvature learnt from the network, 2D: [b, h, w]; 3D: [b, h, w, d]
    :param k2: second curvature learnt from the network, 2D: [b, h, w]; 3D: [b, h, w, d]
    :return: P, propagation matrix, 2D: [b, h, w, 3, 3]; 3D: [b, h, w, d, 4, 4]
    """
    # 2d case
    if len(k1.size()) == 3:
        b, h, w = k1.size()
        skt = torch.sin(k1 * step_len)
        ckt = torch.cos(k1 * step_len)
        # construct the propagation matrix
        P = torch.zeros(b, h, w, 3, 3)
        P[:, :, :, 0, 0] = torch.ones(b, h, w)
        P[:, :, :, 0, 1] = torch.div(skt, k1)
        P[:, :, :, 0, 2] = torch.div(1.0 - ckt, k1)
        P[:, :, :, 1, 1] = ckt
        P[:, :, :, 1, 2] = torch.div(skt, k1)
        P[:, :, :, 2, 1] = -skt
        P[:, :, :, 2, 2] = ckt
    # 3d case
    else:
        b, h, w, d = k1.size()
        k = torch.sqrt(k1 ** 2 + k2 ** 2)
        skt = torch.sin(k * step_len)
        ckt = torch.cos(k * step_len)
        # construct the propagation matrix
        P = torch.zeros(b, h, w, d, 4, 4)
        P[:, :, :, :, 0, 0] = torch.ones(b, h, w, d)
        P[:, :, :, :, 0, 1] = torch.div(skt, k)
        P[:, :, :, :, 0, 2] = torch.div(torch.mul(k1, (1.0 - ckt)), k ** 2)
        P[:, :, :, :, 0, 3] = torch.div(torch.mul(k2, (1.0 - ckt)), k ** 2)
        P[:, :, :, :, 1, 1] = ckt
        P[:, :, :, :, 1, 2] = torch.div(torch.mul(k1, skt), k)
        P[:, :, :, :, 1, 3] = torch.div(torch.mul(k2, skt), k)
        P[:, :, :, :, 2, 1] = torch.div(-torch.mul(k1, skt), k)
        P[:, :, :, :, 2, 2] = torch.div(k2 ** 2 + torch.mul(k1 ** 2, ckt), k ** 2)
        P[:, :, :, :, 2, 3] = torch.div(torch.mul(torch.mul(k1, k2), (ckt - 1.0)), k ** 2)
        P[:, :, :, :, 3, 1] = torch.div(-torch.mul(k2, skt), k)
        P[:, :, :, :, 3, 2] = torch.div(torch.mul(torch.mul(k1, k2), (ckt - 1.0)), k ** 2)
        P[:, :, :, :, 3, 3] = torch.div(k1 ** 2 + torch.mul(k2 ** 2, ckt), k ** 2)
    return P


def propagate_frame(frame, step_len, k1, k2=None):
    """
    generate next frame based on current frame and curvature
    :param frame: current parallel transport frame, shape of [b, h, w, 3, 2]
    :param step_len: sample length for single step, int
    :param k1: first curvature learnt from the network, 2D: [b, h, w]; 3D: [b, h, w, d]
    :param k2: second curvature learnt from the network, 2D: [b, h, w]; 3D: [b, h, w, d]
    :return: next frame, shape of [b, h, w, 3, 2]
    """
    # get the P matrix
    P = construct_P(step_len, k1, k2)
    # reshape the matrix, 2d case
    if len(k1.size()) == 3:
        P = P.view(-1, 3, 3)
        frame = frame.view(-1, 3, 2)
    else:
        P = P.view(-1, 4, 4)
        frame = frame.view(-1, 4, 3)
    next_frame = torch.bmm(P, frame)
    # recover the frame shape
    next_frame = next_frame.view(frame.size())
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


if __name__ == '__main__':
    image_frame = torch.randn(4, 64, 64, 64, 4, 3)
    image_k1 = torch.randn(4, 64, 64, 64)
    image_k2 = torch.randn(4, 64, 64, 64)
    propagate_frame(image_frame, 0.5, image_k1, image_k2)
    validate(image_k1[0, 0, 0, 0], image_k2[0, 0, 0, 0], 0.5)
