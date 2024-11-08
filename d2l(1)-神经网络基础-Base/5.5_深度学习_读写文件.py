import os
import torch
from torch import nn
from torch.nn import functional as F

path_name = current_path = os.path.abspath(__file__).split("\\")[-2]
print(path_name)

# 加载和保存张量
x = torch.arange(4)
torch.save(x, path_name+'x-file')

x2 = torch.load(path_name+'x-file')
print(x2)

y = torch.zeros(4)
torch.save([x, y],path_name+'x-files')
x2, y2 = torch.load(path_name+'x-files')
print(x2, y2)

mydict = {'x': x, 'y': y}
torch.save(mydict, path_name+'mydict')
mydict2 = torch.load(path_name+'mydict')
print(mydict2)



# 加载和保存模型参数
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)

torch.save(net.state_dict(), path_name+'mlp.params')

clone = MLP()
clone.load_state_dict(torch.load(path_name+'mlp.params'))
clone.eval()

Y_clone = clone(X)
print(Y_clone == Y)