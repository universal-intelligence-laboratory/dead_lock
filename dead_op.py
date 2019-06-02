# 可以通过学习，每个通道可以分别进行死亡的卷积核，c个通道就有c个权重
# 控制卷积核的死亡，可以让网络一开始训练一个有力的大模型，随后进行自动化的剪支：
# 1. 一个精心设计的，不太容易死但是一旦死了就很难复活的卷积核
# 2. 训练的时候固定所有权重，裁剪的时候，网络其他部分不变，仅让这些权重可以学习
# 3. 效果好的话，可以加在超分项目上

import torch
import torch.nn.functional as F
from torch import nn, cuda
from torch.autograd import Variable

class DeadConv2d(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1,
                 groups=1, bias=True):
        super(DeadConv2d, self).__init__()
        self.weights = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        self.weights_1x1 = nn.Parameter(torch.Tensor(out_channels, out_channels, 1,1))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = torch.zeros(out_channels)
        self.dead_lock = F.relu(torch.ones(out_channels)) # 用relu控制稀疏性，也可以用tempered softmax控制？

    def forward(self, input):
        output_inner = F.conv2d(input, self.weights, self.bias, 1,
                        self.padding, self.dilation, self.groups)
        
        
        output_dead = F.conv2d(output_inner, self.weights_1x1, self.dead_lock, 1,
                        0, 1,1)# 这样好像控制不了，必须往更深层深度定制，直接在conv的bias后面加一层可以控制何时死亡，稀疏性的bias
        
        
        return output_dead

conv = DeadConv2d(in_channels = 3, out_channels = 3, kernel_size=[3,3])
a = Variable(torch.ones([1,3,10,10]), requires_grad=True)
b = torch.mean(conv(torch.mean(conv(a)) * conv(a)))
b.backward()
print(a.grad)
