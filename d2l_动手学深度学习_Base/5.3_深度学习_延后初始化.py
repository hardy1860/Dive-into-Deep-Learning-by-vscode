"""
检查输入样本格式后在初始化参数大小（shape）
"""

import torch
from torch import nn
from d2l import torch as d2l

net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))

print(net[0].weight)

X = torch.rand(2, 20)
net(X)

print(net[0].weight.shape)

@d2l.add_to_class(d2l.Module)  #@save
def apply_init(self, inputs, init=None):
    self.forward(*inputs)
    if init is not None:
        self.net.apply(init)