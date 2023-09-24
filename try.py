import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"

# 创建一个tensor
x = torch.Tensor([1.0, 2.0])
y = torch.Tensor([1.0, 2.0])

while(True):
        
    # 移动到第一个可见的GPU（即原系统中的GPU 4）
    x = x.cuda()  # 或者 x = x.to('cuda:0')

    # 移动到第二个可见的GPU（即原系统中的GPU 5）
    y = y.cuda()