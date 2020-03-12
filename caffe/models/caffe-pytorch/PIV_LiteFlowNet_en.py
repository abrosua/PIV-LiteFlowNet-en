
import torch
import torch.nn as nn
import torch.legacy.nn as lnn

from functools import reduce
from torch.autograd import Variable

class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

class LambdaMap(LambdaBase):
    def forward(self, input):
        return list(map(self.lambda_func,self.forward_prepare(input)))

class LambdaReduce(LambdaBase):
    def forward(self, input):
        return reduce(self.lambda_func,self.forward_prepare(input))


PIV_LiteFlowNet_en = nn.Sequential( # Sequential,
	nn.Conv2d(3,32,(7, 7),(1, 1),(3, 3)),
	nn.ReLU(),
	nn.ReLU(),
	nn.Conv2d(32,32,(3, 3),(2, 2),(1, 1)),
	nn.ReLU(),
	nn.ReLU(),
	nn.Conv2d(32,32,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.ReLU(),
	nn.Conv2d(32,32,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.ReLU(),
	nn.Conv2d(32,64,(3, 3),(2, 2),(1, 1)),
	nn.ReLU(),
	nn.ReLU(),
	nn.Conv2d(64,64,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.ReLU(),
	nn.Conv2d(64,96,(3, 3),(2, 2),(1, 1)),
	nn.ReLU(),
	nn.ReLU(),
	nn.Conv2d(96,96,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.ReLU(),
	nn.Conv2d(96,128,(3, 3),(2, 2),(1, 1)),
	nn.ReLU(),
	nn.ReLU(),
	nn.Conv2d(128,192,(3, 3),(2, 2),(1, 1)),
	nn.ReLU(),
	nn.ReLU(),
	nn.ReLU(),
	nn.Conv2d(49,128,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(128,64,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(64,32,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(32,2,(3, 3),(1, 1),(1, 1)),
	nn.Conv2d(386,128,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(128,64,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(64,32,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(32,2,(3, 3),(1, 1),(1, 1)),
	nn.Conv2d(195,128,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(128,64,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(64,64,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(64,32,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(32,32,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(32,9,(3, 3),(1, 1),(1, 1)),
)