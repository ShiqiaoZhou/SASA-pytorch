import numpy as np
import torch
a = torch.rand(4,3,1,1)
print(a[:2,-3:,:,:].shape)
print(a[:2,-1:,:,:])
print(a)