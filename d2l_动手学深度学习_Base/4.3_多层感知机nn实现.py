import torch
from torch import nn
from d2l import torch as d2l


# 设置网络模块顺序
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784,256),
                    nn.ReLU(),
                    nn.Linear(256,10))

def init_weight(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weight)

# batch-分批训练批大小、学习率、epoch-整个训练集的训练轮数
batch_size, lr, num_epochs = 256, 0.1, 10

loss = nn.CrossEntropyLoss(reduction='none')

trainer = torch.optim.SGD(net.parameters(),lr=lr)

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

d2l.predict_ch3(net, test_iter)
d2l.plt.show()