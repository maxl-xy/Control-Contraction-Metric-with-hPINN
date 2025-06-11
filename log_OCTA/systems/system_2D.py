import torch

num_dim_x = 2
num_dim_control = 1

def f_func(x):
    # x: bs x n x 1
    # f: bs x n x 1
    bs = x.shape[0]

    x, v = [x[:,i,0] for i in range(num_dim_x)]
    f = torch.zeros(bs, num_dim_x, 1).type(x.type())
    f[:, 0, 0] = v
    f[:, 1, 0] = 0
    return f

def DfDx_func(x):
    raise NotImplemented('NotImplemented')

def B_func(x):
    bs = x.shape[0]
    B = torch.zeros(bs, num_dim_x, num_dim_control).type(x.type())

    B[:, 1, 0] = 1
    return B

def DBDx_func(x):
    raise NotImplemented('NotImplemented')
