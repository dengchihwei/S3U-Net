# -*- coding = utf-8 -*-
# @File Name : plot
# @Date : 2024/7/1 10:05
# @Author : zhiweideng
# @E-mail : zhiweide@usc.edu


from matplotlib import pyplot as plt

x = [0.2, 0.4, 0.6, 0.8, 1.0]
AUC = [0.96183, 0.97963, 0.98261, 0.98295, 0.98367]
ACC = [0.96028, 0.96339, 0.96770, 0.96786, 0.96891]
DICE = [0.76269, 0.80148, 0.81707, 0.81803, 0.81793]
SENS = [0.73328, 0.84677, 0.82919, 0.83027, 0.80342]
SPEC = [0.98230, 0.97481, 0.98114, 0.98122, 0.98495]
LACC = [0.76860, 0.79098, 0.80313, 0.80373, 0.80921]

plt.plot(x, AUC, label='AUC')
plt.plot(x, ACC, label='ACC')
plt.plot(x, DICE, label='DICE')
plt.plot(x, SENS, label='SENS')
plt.plot(x, SPEC, label='SPEC')
plt.plot(x, LACC, label='LACC')
plt.legend()
plt.show()

