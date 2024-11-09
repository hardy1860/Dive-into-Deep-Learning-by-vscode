"""
（一） 引言
计算机视觉研究人员会告诉一个诡异事实
    ————推动领域进步的是数据特征，而不是学习算法。

Dropout、ReLU和预处理是提升计算机视觉任务性能的其他关键步骤。

（二） 核方法—————2001年
    1. 方法: 计算卷积核函数和池化核函数
        picture --> 人工特征提取 --> SVM支持向量机
    2. 特点: 过去核方法和神经网络十年一轮换 CPU or GPU

（三） 神经网络————2010年
    1. 数据集: ImageNet彩色分类图片
    2. 较LeNet主要改进: 丢弃法Dropout、ReLU激活函数、MaxPooling
        picture --> CNN学习特征 --> Softmax回归
    3. 特征提取~数据增强！！！
    4. 更大、更深的LeNet

Question:
    1. 后面为什么两个4096的全连接
        因为前面卷积层抽的不够，所以后面要深一点网络

nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding)
nn.Conv2d(输入通道数    , 输出通道数     , 核尺寸     , 步幅  , 填充) 
"""

import torch
from torch import nn
from d2l import torch as d2l

net = nn.Sequential(
    # 这里使用一个11*11的更大窗口来捕捉对象。
    # 同时，步幅为4，以减少输出的高度和宽度。
    # 另外，输出通道的数目远大于LeNet
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),

    # 使用三个连续的卷积层和较小的卷积窗口。
    # 除了最后的卷积层，输出通道的数量进一步增加。
    # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),

    # 拉成一维来丢人全连接
    nn.Flatten(),
    # 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过拟合
    nn.Linear(6400, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    # 最后是输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
    nn.Linear(4096, 10)
    )


# 检测每层间矩阵是否前后匹配
X = torch.randn(1, 1, 224, 224)
for layer in net:
    X=layer(X)
    print(layer.__class__.__name__,'output shape:\t',X.shape)


"""
训练网络
"""
batch_size = 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

lr, num_epochs = 0.01, 10
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

d2l.plt.show()