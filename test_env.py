import torch

x =torch.tensor((1,2,3))

M = torch.diag(x)

y = torch.diagonal(M)

pass