# MPL

import torch
from torch import nn
from d2l import torch as d2l


batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)


# 初始化模型参数
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]


# ReLu-激活函数(自己写)
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X,a)


# 模型
def net(X):
    X = X.reshape((-1,num_inputs))
    # '@' 表示 矩阵乘法
    H = relu(X @ W1 + b1)
    return (H @ W2 + b2)


# softmax交叉熵-损失函数
loss = nn.CrossEntropyLoss(reduction='none')


# 训练模型
num_epochs, lr = 10, 0.1
# 设置优化方法(迭代更新器)
updater = torch.optim.SGD(params, lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)


d2l.predict_ch3(net, test_iter)
d2l.plt.show()
