# nvidia-smi
# Thu Sep 26 20:46:32 2024       
# +-----------------------------------------------------------------------------------------+
# | NVIDIA-SMI 552.22                 Driver Version: 552.22         CUDA Version: 12.4     |
# |-----------------------------------------+------------------------+----------------------+
# | GPU  Name                     TCC/WDDM  | Bus-Id          Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
# |                                         |                        |               MIG M. |
# |=========================================+========================+======================|
# |   0  NVIDIA GeForce RTX 4080      WDDM  |   00000000:01:00.0  On |                  N/A |
# |  0%   38C    P8              4W /  320W |     873MiB /  16376MiB |      0%      Default |
# |                                         |                        |                  N/A |
# +-----------------------------------------+------------------------+----------------------+

# +-----------------------------------------------------------------------------------------+
# | Processes:                                                                              |
# |  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
# |        ID   ID                                                               Usage      |
# |=========================================================================================|
# |    0   N/A  N/A      6688    C+G   ...09.0.1518.78.x64\msedgewebview2.exe      N/A      |
# |    0   N/A  N/A      6740    C+G   ...p App\AcWebBrowser\AcWebBrowser.exe      N/A      |
# |    0   N/A  N/A     10160    C+G   ...on\129.0.2792.52\msedgewebview2.exe      N/A      |
# |    0   N/A  N/A     10280    C+G   C:\Windows\explorer.exe                     N/A      |
# |    0   N/A  N/A     12640    C+G   ...nt.CBS_cw5n1h2txyewy\SearchHost.exe      N/A      |
# |    0   N/A  N/A     12664    C+G   ...2txyewy\StartMenuExperienceHost.exe      N/A      |
# |    0   N/A  N/A     12972    C+G   ...5\extracted\runtime\WeChatAppEx.exe      N/A      |
# |    0   N/A  N/A     15708    C+G   ...t.LockApp_cw5n1h2txyewy\LockApp.exe      N/A      |
# |    0   N/A  N/A     16864    C+G   ...oogle\Chrome\Application\chrome.exe      N/A      |
# |    0   N/A  N/A     18192    C+G   ...\bin-7.2.0\Nutstore.WindowsHook.exe      N/A      |
# |    0   N/A  N/A     18760    C+G   ...ekyb3d8bbwe\PhoneExperienceHost.exe      N/A      |
# |    0   N/A  N/A     19204    C+G   D:\VSCode\Code.exe                          N/A      |
# |    0   N/A  N/A     20224    C+G   ...GeForce Experience\NVIDIA Share.exe      N/A      |
# |    0   N/A  N/A     20456    C+G   ...GeForce Experience\NVIDIA Share.exe      N/A      |
# |    0   N/A  N/A     20592    C+G   ...CBS_cw5n1h2txyewy\TextInputHost.exe      N/A      |
# |    0   N/A  N/A     24128    C+G   ..._x64__w2gh52qy24etm\SonicRadar3.exe      N/A      |
# |    0   N/A  N/A     24512    C+G   ...x64__w2gh52qy24etm\SonicStudio3.exe      N/A      |
# |    0   N/A  N/A     27268    C+G   ...tstore\bin-7.2.0\NutstoreClient.exe      N/A      |
# |    0   N/A  N/A     27608    C+G   ... Desktop App\AutodeskDesktopApp.exe      N/A      |
# |    0   N/A  N/A     27856    C+G   ...crosoft\Edge\Application\msedge.exe      N/A      |
# |    0   N/A  N/A     28424    C+G   ...5n1h2txyewy\ShellExperienceHost.exe      N/A      |
# |    0   N/A  N/A     34084    C+G   ...siveControlPanel\SystemSettings.exe      N/A      |
# +-----------------------------------------------------------------------------------------+

import torch
from torch import nn

print(torch.device('cpu'), torch.device('cuda'), torch.device('cuda:1'))
print(torch.cuda.device_count())


# test
def try_gpu(i=0):  #@save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():  #@save
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

print(try_gpu(), try_gpu(10), try_all_gpus())


# 张量与GPU
x = torch.tensor([1, 2, 3])
print(x.device)

# 储存在GPU上
X = torch.ones(2, 3, device=try_gpu())
print(X)

# 储存在另一块GPU上
# Y = torch.rand(2, 3, device=try_gpu(1))
# print(Y)


# GPU与神经网络
net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_gpu())

print(net(X))
# 让我们确认模型参数存储在同一个GPU上。
print(net[0].weight.data.device)




