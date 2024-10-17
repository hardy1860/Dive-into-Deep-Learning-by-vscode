import torch
import torch.nn.functional as F
from torch import nn


class CenteredLayer(nn.Module):
    """自定义:  无参数的层"""
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()
    
layer = CenteredLayer()
print(layer(torch.FloatTensor([1, 2, 3, 4, 5])))

# 将层添加到复杂网络中
net = nn.Sequential(nn.Linear(8,128), CenteredLayer())
X = torch.rand(4,8)
Y = net(X)
print(Y.mean())
print(net)



class MyLinear(nn.Module):
    """自定义层: 带参数的层"""
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))

    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)

linear = MyLinear(5, 3)
print(linear(torch.rand(2, 5)))

net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
print(net(torch.rand(2, 64)))


class updown_dim(nn.Module):
    def __init__(self, i_units, j_units, k_units):
        """
        Question1: 设计一个接受输入并计算张量降维的层，它返回yk = sum_ij W_ijk*xi*xj
        
        net = updown_dim(i, j, k), i=j

        output = net(X)

        X.shape = (i)
        """
        super().__init__()
        self.weight = nn.Parameter(torch.randn(k_units, i_units, j_units))

    def forward(self, X):
        Y = torch.matmul(self.weight, X)
        return torch.matmul(Y, X)
    
net = updown_dim(8, 8, 7)
print(net.weight.shape)
print(net(torch.rand(8)))





