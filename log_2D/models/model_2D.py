import torch
from torch import nn
from torch.autograd import grad
import numpy as np

effective_dim_start = 1
effective_dim_end = 2

class U_FUNC(nn.Module):
    """docstring for U_FUNC."""

    def __init__(self, model_u_w1, model_u_w2, num_dim_x, num_dim_control):
        super(U_FUNC, self).__init__()
        self.model_u_w1 = model_u_w1
        self.model_u_w2 = model_u_w2
        self.num_dim_x = num_dim_x
        self.num_dim_control = num_dim_control

    def forward(self, x, xe, uref):
        # x: B x n x 1
        # u: B x m x 1
        bs = x.shape[0]

        # w1_xxref = self.model_u_w1(torch.cat([x,(x-xe)],dim=1).squeeze(-1)).reshape(bs, self.num_dim_control, -1)
        # w1_xx = self.model_u_w1(torch.cat([x,x],dim=1).squeeze(-1)).reshape(bs, self.num_dim_control, -1)
        # u = w1_xxref - w1_xx + uref

        w1_xe = self.model_u_w1(torch.cat([x,xe],dim=1).squeeze(-1)).reshape(bs, self.num_dim_control, -1)
        w1_x0 = self.model_u_w1(torch.cat([x,torch.zeros(xe.shape).type(xe.type())],dim=1).squeeze(-1)).reshape(bs, self.num_dim_control, -1)
        u = .8* (w1_xe - w1_x0) + uref
        return u

def get_model(num_dim_x, num_dim_control, w_lb, use_cuda = False):
    model_Wbot = torch.nn.Sequential(
        torch.nn.Linear(1, 128, bias=True),
        torch.nn.Tanh(),
        torch.nn.Linear(128, (num_dim_x-num_dim_control) ** 2, bias=False))

    dim = effective_dim_end - effective_dim_start
    model_W = torch.nn.Sequential(
        torch.nn.Linear(dim, 128, bias=True),
        torch.nn.Tanh(),
        torch.nn.Linear(128, num_dim_x * num_dim_x, bias=False))

    c = 3 * num_dim_x
    model_u_w1 = torch.nn.Sequential(
        torch.nn.Linear(2*num_dim_x, 128, bias=True),
        torch.nn.Tanh(),
        torch.nn.Linear(128, num_dim_control, bias=True))

    model_u_w2 = torch.nn.Sequential(
        torch.nn.Linear(2*num_dim_x, 128, bias=True),
        torch.nn.Tanh(),
        torch.nn.Linear(128, num_dim_control, bias=True))

    if use_cuda:
        model_W = model_W.cuda()
        model_Wbot = model_Wbot.cuda()
        model_u_w1 = model_u_w1.cuda()
        model_u_w2 = model_u_w2.cuda()

    def W_func(x):
        bs = x.shape[0]
        x = x.squeeze(-1)

        W = model_W(x[:,effective_dim_start:effective_dim_end]).view(bs, num_dim_x, num_dim_x)
        Wbot = model_Wbot(torch.ones(bs, 1).type(x.type())).view(bs, num_dim_x-num_dim_control, num_dim_x-num_dim_control)
        W[:, 0:num_dim_x-num_dim_control, 0:num_dim_x-num_dim_control] = Wbot
        W[:, num_dim_x-num_dim_control::, 0:num_dim_x-num_dim_control] = 0

        # W = model_W(x[:, effective_dim_start:effective_dim_end]).view(bs, num_dim_x, num_dim_x)

        W = W.transpose(1,2).matmul(W)
        W = W + w_lb * torch.eye(num_dim_x).view(1, num_dim_x, num_dim_x).type(x.type())
        return W


    u_func = U_FUNC(model_u_w1, model_u_w2, num_dim_x, num_dim_control)

    return model_W, model_Wbot, model_u_w1, model_u_w2, W_func, u_func
