import torch
from IPython import display
from d2l import torch as d2l

import numpy as np

X = torch.normal(0, 1, (2, 5))
print('X---',X)

X_exp = torch.exp(X)
print('X_exp---',X_exp)

partition = X_exp.sum(1, keepdim=True)
print('partition---',partition)

print('softmax---',X_exp / partition)

partition1 = torch.tensor([[partition[0,0]]*5,[partition[1,0]]*5 ])
print('----',partition1)

print('手动广播---',X_exp /partition1)