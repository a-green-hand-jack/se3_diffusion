
#%%
# 研究 np.interp 的功能
import numpy as np
     
x = np.array([1, 2, 3, 4]) * 10
# y = np.array([10, 20, 30, 40])
y = np.exp(x)
x_new = 3.5 * 10
y_gt = np.exp(x_new)
     
y_new = np.interp(x_new, x, y)  # 计算 x_new 对应的 y 值
print(y_new)  # 输出: 25.0
print(y_gt)