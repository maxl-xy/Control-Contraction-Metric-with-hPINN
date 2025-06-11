import torch
from torch import nn

# Adjust these indices to match the "effective" state dimensions for your system
effective_dim_start = 4
effective_dim_end = 12

class U_FUNC(nn.Module):
    def __init__(self, model_u_w1, model_u_w2, num_dim_x, num_dim_control):
        super(U_FUNC, self).__init__()
        self.model_u_w1 = model_u_w1
        self.model_u_w2 = model_u_w2
        self.num_dim_x = num_dim_x
        self.num_dim_control = num_dim_control

    def forward(self, x, xe, uref):
        bs = x.shape[0]
        # Adjust slicing for your 12D state
        x_eff = x[:, effective_dim_start:effective_dim_end, :]
        xe_eff = (x - xe)[:, effective_dim_start:effective_dim_end, :]
        w1 = self.model_u_w1(torch.cat([x_eff, xe_eff], dim=1).squeeze(-1)).reshape(bs, -1, self.num_dim_x)
        w2 = self.model_u_w2(torch.cat([x_eff, xe_eff], dim=1).squeeze(-1)).reshape(bs, self.num_dim_control, -1)
        u = w2.matmul(torch.tanh(w1.matmul(xe))) + uref
        return u

def get_model(num_dim_x, num_dim_control, w_lb, use_cuda=False):
    dim = effective_dim_end - effective_dim_start
    model_Wbot = nn.Sequential(
        nn.Linear(dim-num_dim_control, 128, bias=True),
        nn.Tanh(),
        nn.Linear(128, (num_dim_x-num_dim_control) ** 2, bias=False)
    )
    model_W = nn.Sequential(
        nn.Linear(dim, 128, bias=True),
        nn.Tanh(),
        nn.Linear(128, num_dim_x * num_dim_x, bias=False)
    )
    c = 3 * num_dim_x
    model_u_w1 = nn.Sequential(
        nn.Linear(2*dim, 128, bias=True),
        nn.Tanh(),
        nn.Linear(128, c*num_dim_x, bias=True)
    )
    model_u_w2 = nn.Sequential(
        nn.Linear(2*dim, 128, bias=True),
        nn.Tanh(),
        nn.Linear(128, num_dim_control*c, bias=True)
    )
    if use_cuda:
        model_W = model_W.cuda()
        model_Wbot = model_Wbot.cuda()
        model_u_w1 = model_u_w1.cuda()
        model_u_w2 = model_u_w2.cuda()

    def W_func(x):
        bs = x.shape[0]
        x = x.squeeze(-1)
        W = model_W(x[:, effective_dim_start:effective_dim_end]).view(bs, num_dim_x, num_dim_x)
        Wbot = model_Wbot(x[:, effective_dim_start:effective_dim_end-num_dim_control]).view(bs, num_dim_x-num_dim_control, num_dim_x-num_dim_control)
        W[:, 0:num_dim_x-num_dim_control, 0:num_dim_x-num_dim_control] = Wbot
        W[:, num_dim_x-num_dim_control:, 0:num_dim_x-num_dim_control] = 0
        W = W.transpose(1,2).matmul(W)
        W = W + w_lb * torch.eye(num_dim_x).view(1, num_dim_x, num_dim_x).type(x.type())
        return W

    u_func = U_FUNC(model_u_w1, model_u_w2, num_dim_x, num_dim_control)
    return model_W, model_Wbot, model_u_w1, model_u_w2, W_func, u_func