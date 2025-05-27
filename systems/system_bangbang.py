import torch

num_dim_x = 2
num_dim_control = 1

def f_func(x):
    # x: [batch, n, 1]
    dx = torch.zeros_like(x)
    dx[:, 0, 0] = x[:, 1, 0]  # dx/dt = v
    dx[:, 1, 0] = 0           # dv/dt = u (handled by B_func)
    return dx

def B_func(x):
    # x: [batch, n, 1]
    B = torch.zeros(x.shape[0], 2, 1).type(x.type())
    B[:, 1, 0] = 1  # Only velocity is actuated
    return B