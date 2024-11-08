# 线性回归：y = sum wi*xi
import math
import numpy as np
import torch
from d2l import torch as d2l


# 【3.1.2】 矢量化加速
n = 10000
a = torch.ones([n])
b = torch.ones([n])

# for循环计算加法
c = torch.zeros([n])
timer = d2l.Timer()
for i in range(n):
    c[i] = a[i] + b[i]
print(f'{timer.stop():.9f} sec')

# 矢量化 + 运算
timer.start()
d = a + b
print(f'{timer.stop():.9f} sec')

