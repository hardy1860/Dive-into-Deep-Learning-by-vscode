# multilayer perceptron (MPL)
"""
one of the layers :
+-----------+    +--------------+    +--------------+    +----------+
|   input   |    |   connect    |    |  activation  |    |  output  |
|     X     |--->|    W*X+b     |--->|   function:  |--->|    O     |
|   (n*1)   |    | W(n*w) b(w*1)|    | sigma(W*X+b) |    |  (w*1)   |
+-----------+    +--------------+    +--------------+    +----------+
【思考？】通过对 W 的内容来做文章，达到全连接(full connect)等。
"""

import torch
from d2l import torch as d2l

"""激活函数"""
# 1. ReLU
# ReLU函数有许多变体，包括参数化ReLU（Parameterized ReLU，pReLU）
# 该变体为ReLU添加了一个线性项，因此即使参数是负的，某些信息仍然可以通过
x = torch.arange(-5.0,5.0,0.1,requires_grad=True)
y = torch.relu(x)
d2l.plt.figure(figsize=(15,5))
d2l.plt.subplot(2,3,1)
d2l.plot(x.detach(),y.detach(),'x','ReLU(x)',figsize=(5,2.5))
# ReLU函数的导数
y.backward(torch.ones_like(x), retain_graph=True)
d2l.plt.subplot(2,3,4)
d2l.plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))

# 2. sigmoid
# 隐藏层中很少使用sigmoid，一般用简单易训练的ReLU在
# 循环神经网络中，可以利用sigmoid单元来控制时序信息流的架构
y = torch.sigmoid(x)
d2l.plt.subplot(2,3,2)
d2l.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
# sigmoid函数的导数
# 清除以前的梯度
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plt.subplot(2,3,5)
d2l.plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))

# 3. tanh
y = torch.tanh(x)
d2l.plt.subplot(2,3,3)
d2l.plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))
# tanh函数的导数
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plt.subplot(2,3,6)
d2l.plot(x.detach(), x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))

d2l.plt.show()
