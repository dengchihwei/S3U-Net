# -*- coding = utf-8 -*-
# @File Name : visualize
# @Date : 2023/9/20 11:23
# @Author : zhiweideng
# @E-mail : zhiweide@usc.edu


import torch
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_erosion, binary_dilation


def smooth(ves, s=1):
    if isinstance(ves, torch.Tensor):
        sm = gaussian_filter(ves.data.numpy(), sigma=s)
        smoothed = torch.Tensor(sm)
    else:
        smoothed = gaussian_filter(ves, sigma=s)
    return smoothed


THRESHOLD = 0.006
PATH_POINTS_NUM = 20
# load data
idx = 9
res = torch.load('/Users/zhiweideng/Desktop/NICR/VesselAnalysis/ConnectVessel/PTF_Analysis/res_{}.pt'.format(idx))[0]
path = torch.load('/Users/zhiweideng/Desktop/NICR/VesselAnalysis/ConnectVessel/PTF_Analysis/path_{}.pt'.format(idx))
path = path.view(path.size(0), path.size(1), PATH_POINTS_NUM, 2)
# generate the mask using fixed threshold
mask = res > THRESHOLD
spacing = 128.0 / 127.0


# loop over to update the response intensities
update_ratio = 0.5
count_res = torch.zeros_like(res)
updated_res = torch.zeros_like(res)
for i in range(PATH_POINTS_NUM):
    curr_path = torch.multiply(path[:, :, i], mask.unsqueeze(-1))
    curr_path_idx = torch.round((curr_path + 0.5) / spacing).long().view(-1, 2)
    curr_path_idx = torch.where(curr_path_idx > 127, torch.tensor(127), curr_path_idx)
    curr_path_idx = torch.where(curr_path_idx < 0, torch.tensor(0), curr_path_idx)
    curr_vals = torch.zeros_like(res)
    curr_vals[curr_path_idx[:, 0], curr_path_idx[:, 1]] = torch.multiply(mask, res).view(-1)
    curr_vals = torch.multiply(curr_vals.view(128, 128), ~mask)
    count_res += curr_vals > 0.0
    updated_res += curr_vals * 1.0

updated_res = torch.div(updated_res, count_res + 1)
updated_res = res + updated_res

vesselness = smooth(res, s=0.60)
v_threshed = (vesselness >= 0.008)
v_threshed = binary_erosion(v_threshed, np.ones((1, 1)))
v_threshed = remove_small_objects(v_threshed, min_size=40)
plt.figure()
plt.imshow(v_threshed)
plt.colorbar()

vesselness = smooth(updated_res, s=0.60)
v_threshed = (vesselness >= 0.008)
v_threshed = binary_erosion(v_threshed, np.ones((1, 1)))
v_threshed = remove_small_objects(v_threshed, min_size=40)
plt.figure()
plt.imshow(v_threshed)
plt.colorbar()

plt.show()



# # x, y = torch.randint(0, 128, (2,))
# x, y = 12, 74
# # res[x, y] = 0.1
# point_path1 = path[x, y].view(20, 2)
# point_path2 = path[x-1, y-1].view(20, 2)
# point_path3 = path[x+1, y+1].view(20, 2)
# point_path4 = path[x+2, y].view(20, 2)
# point_path5 = path[x, y+1].view(20, 2)
# # print(dirs[x, y])
# # print(k[x, y].mean())
# # for i in range(60):
# #     p = point_path[i]
# #     res[int(p[0]), int(p[1])] = 0.005 * i
#
#
# plt.plot(point_path1[:, 1], point_path1[:, 0], color='red')
# plt.plot(point_path2[:, 1], point_path2[:, 0], color='red')
# plt.plot(point_path3[:, 1], point_path3[:, 0], color='red')
# plt.plot(point_path4[:, 1], point_path4[:, 0], color='red')
# plt.plot(point_path5[:, 1], point_path5[:, 0], color='red')
# plt.imshow(res)
# plt.colorbar()
# plt.show()
