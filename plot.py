# -*- coding = utf-8 -*-
# @File Name : plot
# @Date : 2024/7/1 10:05
# @Author : zhiweideng
# @E-mail : zhiweide@usc.edu


from matplotlib import pyplot as plt

plt.style.use('ggplot')

x = [20, 40, 60, 80, 100]
AUC = [0.96183, 0.97963, 0.98261, 0.98295, 0.98367]
ACC = [0.96028, 0.96339, 0.96770, 0.96786, 0.96891]
SENS = [0.73328, 0.84677, 0.82919, 0.83027, 0.80342]
SPEC = [0.98230, 0.97481, 0.98114, 0.98122, 0.98495]
LACC = [0.76860, 0.79098, 0.80313, 0.80373, 0.80921]

DRIVE_DICE = [0.78269, 0.80148, 0.81707, 0.81803, 0.82393]
DRIVE_DICE_UNET = [0.73613, 0.76624, 0.79265, 0.79674, 0.80250]

# baseline
SMILE_DICE = [0.7624, 0.7785, 0.7843, 0.7937, 0.8036]
SMILE_DICE_UNET = [0.6932, 0.7329, 0.7698, 0.7714, 0.7835]


LSA_DICE = [0.5387900810785982, 0.5412307307156089, 0.5789808663261177, 0.585274784638835, 0.5902939218409674]
LSA_DICE_UNET = [0.46577758696036933, 0.50470326557369625, 0.5469192312329618, 0.5584306362227491, 0.5764333020208783]


LSA_UN = [0.5145, 0.5145, 0.5145, 0.5145, 0.5145]
DRIVE_UN = [0.7721, 0.7721, 0.7721, 0.7721, 0.7721]
SMILE_UN = [0.7431, 0.7431, 0.7431, 0.7431, 0.7431]

# plt.plot(x, AUC, label='AUC')
# plt.plot(x, ACC, label='ACC')
# plt.plot(x, DRIVE_UN, '--', label='DRIVE-Unsup.')
# plt.plot(x, DRIVE_DICE_UNET, '--o', label='DRIVE-Scratch')
# plt.plot(x, DRIVE_DICE, '--o', label='DRIVE-Pre-trained')

plt.plot(x, SMILE_UN, '--', label='SMILE(S3U-Net)')
plt.plot(x, SMILE_DICE_UNET, '--o', label='SMILE(Random init)')
plt.plot(x, SMILE_DICE, '--o', label='SMILE(Pre-trained)')



# plt.plot(x, SENS, label='SENS')
# plt.plot(x, SPEC, label='SPEC')
# plt.plot(x, LACC, label='LACC')

plt.plot(x, LSA_UN, '--', label='BBMRA(S3U-Net)')
plt.plot(x, LSA_DICE_UNET, '--o', label='BBMRA(Random init)')
plt.plot(x, LSA_DICE, '--o', label='BBMRA(Pre-trained)')


plt.legend(loc='best')
plt.xlabel('Annotations Used(%)')
plt.ylabel('Dice Score')
plt.savefig('test.svg', bbox_inches='tight')
plt.show()


# lambdas = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
# lambdas_dices = [0.4825, 0.4879, 0.5037, 0.5033, 0.5145, 0.5011]
#
# ms = ['16', '32', '64', '128', '256']
# ms_dices = [0.3371, 0.4858, 0.4925, 0.5145, 0.5149]
#
#
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 2))
# plt.subplots_adjust(hspace=0.3)
# ax1.plot(lambdas, lambdas_dices, '--o', color='salmon')
# ax1.set_ylim(0.475, 0.525)
# ax1.set_xticks(lambdas)
# ax1.set_ylabel('Dice Score')
# ax1.set_xlabel('lambda')
# for i, j in zip(lambdas, lambdas_dices):
#     ax1.annotate(str(j), xy=(i-0.2, j+0.002))
#
# ax2.plot(ms, ms_dices, '--o', color='darkseagreen')
# ax2.set_xticks(ms)
# ax2.set_ylabel('Dice Score')
# ax2.set_ylim(0.325, 0.545)
# ax2.set_xlabel('m')
# for i, j in zip(range(6), ms_dices):
#     i = float(i)
#     ax2.annotate(str(j), xy=(i-0.2, j+0.002))
#
# plt.savefig('test.svg', bbox_inches='tight')

# plt.show()



