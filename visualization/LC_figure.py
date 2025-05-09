# -*- coding = utf-8 -*-
# @File Name : LC_figure
# @Date : 2024/12/10 19:42
# @Author : zhiweideng
# @E-mail : zhiweide@usc.edu


import numpy as np
import nibabel as nib
import SimpleITK as sitk
from matplotlib import pyplot as plt

path = '/Users/zhiweideng/Desktop/NICR/VesselAnalysis/7T_MRA/organized/train/M003/M003_TOF_MASKED.nii.gz'




x, y, z = 200, 100, 160
image = nib.load(path).get_fdata()
print(image.shape)
image = sitk.GetArrayFromImage(sitk.ReadImage(path))
print(image.shape)
image_patch = image[x-32:x+32, y-32:y+32, z-32:z+32]
plt.imshow(image_patch[:, :, 12])
plt.show()






# def contrast_test():
#     # Test for 3D images
#     import numpy as np
#     # import nibabel as nib
#     # path = '/Users/dengzhiwei/Desktop/NICR/MSA_Data/seletected_data/old/A007/TOF/'
#     # file = path + '8_fl_tof_tra_p3_wholebrain_04iso.nii.gz'
#     # i, j, k = 200, 100, 160
#     # image_patch = nib.load(file).get_fdata()[i-32:i+32, j-32:j+32, k-32:k+32]
#     # image_patch = torch.from_numpy(image_patch)
#     # max_val, min_val = image_patch.max(), image_patch.min()
#     # image_patch = 1.0 * (image_patch - min_val) / (max_val - min_val)
#     # image_patch = image_patch.unsqueeze(0).unsqueeze(0).float()
#     #
#     # test_radius = torch.zeros_like(image_patch)
#     # test_radius[:, 0, :, :, :] = 1.0
#     # img_local_contrast = calc_local_contrast(image_patch, test_radius, 16, 5)
#     # max_val, min_val = img_local_contrast.max(), img_local_contrast.min()
#     # img_local_contrast = 1.0 * (img_local_contrast - min_val) / (max_val - min_val)
#     #
#     # _x, _y = np.arange(64),  np.arange(64)
#     # _xx, _yy = np.meshgrid(_x, _y)
#     # x, y = _xx.ravel(), _yy.ravel()
#     #
#     # fig = plt.figure(figsize=(12, 5))
#     # ax1 = fig.add_subplot(121, projection='3d')
#     # ax2 = fig.add_subplot(122, projection='3d')
#     # index = 10
#     # original_image = image_patch[0, 0, :, :, index].numpy().ravel()
#     # contrast_image = img_local_contrast[0, 0, :, :, index].numpy().ravel()
#     #
#     # bottom = np.zeros_like(original_image)
#     # ax1.bar3d(x, y, bottom, 1, 1, original_image, shade=True)
#     # ax2.bar3d(x, y, bottom, 1, 1, contrast_image, shade=True)
#     #
#     # plot_slice(image_patch[0, 0, :, :, index], None, None)
#     # plot_slice(img_local_contrast[0, 0, :, :, index], None, None)
#     # plt.show()
#
#     # Test for 2D images
#     from PIL import Image
#     from scipy.ndimage import gaussian_filter
#     path = '/Users/dengzhiwei/Desktop/NICR/EyeVessel/DRIVE/test/images/02_test.tif'
#     mask_path = '/Users/dengzhiwei/Desktop/NICR/EyeVessel/DRIVE/test/mask/02_test_mask.gif'
#     image = np.array(Image.open(path))[..., 1] / 255.0
#     mask = torch.from_numpy(np.array(Image.open(mask_path)))
#     # att_path = '/Users/dengzhiwei/Desktop/NICR/drive_results/adaptive_lc/drive_rad_test_naive.npy'
#     # att = np.load(att_path)
#     # test_radius = torch.from_numpy(att[2:3]).unsqueeze(0).float()
#     image = gaussian_filter(1.0 - image, 0.45)
#     image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
#     gradient = torch.gradient(image, dim=(2, 3))
#     gradient = gradient[1]      # + gradient[1]
#     test_radius = torch.zeros_like(image)
#     test_radius[:, 0, :, :] = 2.0
#     img_local_contrast = calc_local_contrast(image, test_radius, 32, 5)
#     print(img_local_contrast.mean())
#
#     plot_slice(1.0 - image[0, 0], None, None)
#     plot_slice(img_local_contrast[0, 0], None, None)
#     # plot_slice(gradient[0, 0], None, None)
#     plot_slice(test_radius[0, 0], None, None)
#     plt.show()