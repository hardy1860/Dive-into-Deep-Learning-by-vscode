import torch
from IPython import display
from d2l import torch as d2l

X = torch.arange(5,dtype=torch.float)
A = torch.normal(2,0.5,(3,5))
B = torch.normal(3,0.5,(5,4))
print(X,'\n',A,'\n',B)

# 1. 矩阵向量积测试
print(torch.mv(A,X))
# print(torch.mv(A,B))

# 2. 矩阵矩阵积测试
# print(torch.mm(A,X))
print(torch.mm(A,B))

"""
1. 结论：
    R(n*0*0),一维矩阵为：向量
    R(n*m*q),多维矩阵为：矩阵
    
    ---向量向量积(点积dot)，矩阵向量积(mv)，矩阵矩阵积(mm)

2. mv,生成的向量须加【dtype=torch.float】改变类型后计算

"""

