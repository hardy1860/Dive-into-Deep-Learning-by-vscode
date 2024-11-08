"""
训练模型时试图找到一个能够尽可能拟合训练数据的函数。 但是如果它执行地“太好了”，
而不能对看不见的数据做到很好泛化，就会导致过拟合。这种情况正是我们想要避免或控制的。 
++++++++++++++++++++++++++++++++++++++++++++++++
|  泛化能力 |         一般判断方法              |
++++++++++++++++++++++++++++++++++++++++++++++++
|   过拟合  |  train error << validation error |
+----------++---------------------------------++
|   欠拟合  |  train error ≈ validation error  |  
++++++++++++++++++++++++++++++++++++++++++++++++

1. 稀疏数据如何选取合适的`test_iter` ?
    --------K折交叉验证/S折交叉验证(K/S-fold cross validation)《统计学习方法》

"""

# 多项式回归(三阶)
import math
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l


max_degree = 20     # 多项式最大阶数
n_train, n_test = 100, 100
true_w = np.zeros(max_degree)
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

features = np.random.normal(size=(n_train + n_test, 1))
np.random.shuffle(features)
poly_features = np.power(features, np.arange(max_degree).reshape(1,-1))
for i in range(max_degree):
    # gamma(n)=(n-1)!
    poly_features[:,i] /= math.gamma(i + 1)
# labels的维度:(n_train + n_test,)
labels = np.dot(poly_features, true_w)
labels += np.random.normal(scale=0.1, size=labels.shape)
# print(features.shape,poly_features.shape,labels.shape)
# shape : (200, 1) (200, 20) (200,)

true_w, features, poly_features, labels = [
    torch.tensor(x,dtype=torch.float32) for x in [true_w, features, poly_features, labels]]
# print(features[:2], poly_features[:2, :], labels[:2])


# 损失函数
def evaluate_loss(net, data_iter, loss):
    """评估给定数据集上模型的损失"""
    metric = d2l.Accumulator(2)
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric


# 训练函数
def train(train_features, test_features, train_labels, test_labels, num_epochs=400):
    # 最小二乘损失
    loss = nn.MSELoss(reduce='mean')
    input_shape = train_features.shape[-1]
    # 不设偏置，因为我们在多项式中已经实现了
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels.reshape(-1,1)),batch_size)
    test_iter = d2l.load_array((test_features, test_labels.reshape(-1,1)),
                               batch_size, is_train=False)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                            yscale='log', legend=['train', 'test'])
    for epoch in range(num_epochs):
        d2l.train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch==0 or (epoch + 1)%20==0:
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
                                     evaluate_loss(net, test_iter, loss)))
    print('weight:', net[0].weight.data.numpy())


# 从多项式特征中选择前4个维度，即1,x,x^2/2!,x^3/3!
# 比较好
train(poly_features[:n_train, :4], poly_features[n_train:, :4],
      labels[:n_train], labels[n_train:])


# # 从多项式特征中选择前2个维度，即1和x
# 欠拟合
train(poly_features[:n_train, :2], poly_features[n_train:, :2],
      labels[:n_train], labels[n_train:])


# 从多项式特征中选取所有维度
# 过拟合
train(poly_features[:n_train, :], poly_features[n_train:, :],
      labels[:n_train], labels[n_train:], num_epochs=1500)
