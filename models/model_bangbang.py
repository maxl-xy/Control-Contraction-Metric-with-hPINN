import torch
import torch.nn as nn

class SimpleW(nn.Module):
    def __init__(self, num_dim_x):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_dim_x, 8), nn.Tanh(),
            nn.Linear(8, num_dim_x)
        )

    def forward(self, x):
        # x: [batch, n, num_dim_x]
        x_flat = x.view(x.shape[0], -1)
        w = self.net(x_flat)
        # Ensure positive values (e.g., softplus)
        w = torch.nn.functional.softplus(w) + 1e-3
        return torch.diag_embed(w)

class SimpleU(nn.Module):
    def __init__(self, num_dim_x, u_lower, u_upper):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_dim_x, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1)
        )
        self.u_lower = u_lower
        self.u_upper = u_upper

    def forward(self, x, xe, uref):
        u = self.net(x.view(x.shape[0], -1))
        u = torch.clamp(u, self.u_lower, self.u_upper)
        return u.unsqueeze(-1)

def get_model(num_dim_x, num_dim_control, w_lb, use_cuda=False):
    model_W = SimpleW(num_dim_x)
    model_Wbot = nn.Identity()
    model_u_w1 = SimpleU(num_dim_x, -0.5, 0.5)
    model_u_w2 = nn.Identity()
    def W_func(x): return model_W(x)
    def u_func(x, xe, uref): return model_u_w1(x, xe, uref)
    if use_cuda:
        model_W.cuda()
        model_u_w1.cuda()
    return model_W, model_Wbot, model_u_w1, model_u_w2, W_func, u_func