import torch

# ---- Dynamics for 12D slung payload quadrotor ----

# Parameters (keeping these as scalars)
mp = 0.5        # Mass of the payload
mq = 1.63       # Mass of the quadcopter

class Dynamics:
    def __init__(self, x, mp=mp, mq=mq, dt=None, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.dt = dt
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=device)
        else:
            x = x.to(device)
        self.x = x
        self.mp = mp
        self.mq = mq
        self.l0 = torch.tensor(1.0, dtype=x.dtype, device=device)

        self.B_val = None
        self.B_dot_val = None
        self.N_val = None
        self.Mass_val = None
        self.C_val = None

        self.set_state(x)
        self.compute_Dynamics()
        
    def set_state(self, x):
        if not isinstance(x, torch.Tensor):
            self.x = torch.tensor(x, dtype=torch.float32, device=self.device)
        else:
            self.x = x.to(self.device)
        
    def compute_Dynamics(self):
        self.get_B()
        self.get_B_dot()
        self.get_N()
        self.get_MassMatrix()
        self.get_CoriolisMatrix()

    def get_B(self):
        r = self.x[0:2].view(-1)
        denominator = torch.sqrt(1 - torch.dot(r, r))
        B_last_row = -r / denominator  # shape (2,)
        eye2 = torch.eye(2, dtype=self.x.dtype, device=self.x.device)
        self.B_val = torch.cat([eye2, B_last_row.unsqueeze(0)], dim=0)  # shape (3,2)

    def get_B_dot(self):
        r = self.x[0:2].view(-1)      # flatten r to 1D
        r_dot = self.x[2:4].view(-1)  # flatten r_dot to 1D
        denominator = (1 - torch.dot(r, r)) ** (3/2)
        numerator = torch.dot(r, r_dot) * r
        B_dot_last_row = numerator / denominator  # shape (2,)
        zeros_top = torch.zeros((2,2), dtype=self.x.dtype, device=self.x.device)
        self.B_dot_val = torch.cat([zeros_top, B_dot_last_row.unsqueeze(0)], dim=0)  # shape (3,2)

    def get_N(self):
        r = self.x[0:2].reshape(2, 1)
        sqrt_term = torch.sqrt(1 - torch.dot(r.squeeze(), r.squeeze()))
        self.N_val = torch.cat([r, sqrt_term.view(1,1)], dim=0)  # shape (3,1)

    def get_MassMatrix(self):
        dl = self.x[10]
        l = self.l0 + dl
        M11 = (l**2) * self.mp * (self.B_val.t() @ self.B_val)          # (2,2)
        M12 = l * self.mp * self.B_val.t()                                # (2,3)
        M13 = torch.zeros((2,1), dtype=self.x.dtype, device=self.x.device)  # (2,1)
        M21 = l * self.mp * self.B_val                                  # (3,2)
        M22 = (self.mp + self.mq) * torch.eye(3, dtype=self.x.dtype, device=self.x.device)  # (3,3)
        M23 = self.mp * self.N_val                                      # (3,1)
        M31 = torch.zeros((1,2), dtype=self.x.dtype, device=self.x.device)  # (1,2)
        M32 = self.mp * self.N_val.t()                                  # (1,3)
        M33 = torch.tensor([[self.mp]], dtype=self.x.dtype, device=self.x.device)  # (1,1)

        top = torch.cat([M11, M12, M13], dim=1)
        middle = torch.cat([M21, M22, M23], dim=1)
        bottom = torch.cat([M31, M32, M33], dim=1)
        self.Mass_val = torch.cat([top, middle, bottom], dim=0)

    def get_CoriolisMatrix(self):
        x = self.x
        mp = self.mp
        dl = self.x[10]
        l = self.l0 + dl
        ldot = x[11]
        r_dot = x[2:4].reshape(2,1)
        aux = l * self.B_dot_val + ldot * self.B_val  # (3,2)
        C11 = l * mp * (self.B_val.t() @ aux)          # (2,2)
        C12 = torch.zeros((2,3), dtype=self.x.dtype, device=self.x.device)
        C13 = l * mp * (self.B_val.t() @ (self.B_val @ r_dot))  # (2,1)
        C21 = mp * aux                                 # (3,2)
        C22 = torch.zeros((3,3), dtype=self.x.dtype, device=self.x.device)
        C23 = mp * (self.B_val @ r_dot)                  # (3,1)
        C31 = l * mp * (self.N_val.t() @ self.B_dot_val) # (1,2)
        C32 = torch.zeros((1,3), dtype=self.x.dtype, device=self.x.device)
        C33 = torch.zeros((1,1), dtype=self.x.dtype, device=self.x.device)

        top = torch.cat([C11, C12, C13], dim=1)
        middle = torch.cat([C21, C22, C23], dim=1)
        bottom = torch.cat([C31, C32, C33], dim=1)
        self.C_val = torch.cat([top, middle, bottom], dim=0)

# ---- Fx_Gx Calculations ----

class Fx_Gx_Calculations(Dynamics):
    def __init__(self, x, mp=mp, mq=mq, device=None):
        super().__init__(x, mp, mq, device=device)
        self.compute_Fx_Gx()

    def compute_Fx_Gx(self):
        self.get_Fx()
        self.get_Gx()

    def get_Fx(self):
        mp = self.mp
        x = self.x
        B_val = self.B_val
        Mass_val = self.Mass_val
        C_val = self.C_val
        l = self.x[10]
        g_I = torch.tensor([0, 0, 9.81], dtype=x.dtype, device=self.device)

        # Compute Fx using the provided formula
        term1 = torch.cat([(mp*l*torch.matmul(B_val.t(), g_I)).view(2,1), 
                           torch.zeros((3, 1), dtype=x.dtype, device=self.device), 
                           (torch.tensor([0], dtype=x.dtype, device=self.device)).view(1,1)], dim=0)
        
        x_org = torch.cat([
            x[2:4].reshape(2, 1),           # Reshape to (2, 1)
            x[7:10].reshape(3, 1),          # Reshape to (3, 1)
            x[11].reshape(1, 1)             # Reshape to (1, 1)
        ], dim=0)                          # Combine into (6, 1)
        
        term2 = torch.matmul(C_val, x_org)
        F1 = torch.linalg.solve(Mass_val, term1 - term2)
        Fx = torch.cat([F1, x_org], dim=0)
        self.Fx = Fx

    def get_Gx(self):
        Mass_val = self.Mass_val
        term1 = torch.cat([
            torch.zeros((2, 4), dtype=self.x.dtype, device=self.device),
            torch.eye((4), dtype=self.x.dtype, device=self.device)
        ], dim=0)
        Mass_val = 0.5 * (Mass_val + Mass_val.t())
        Mass_val_t = torch.linalg.inv(Mass_val)
        term2 = torch.zeros((6, 4), dtype=self.x.dtype, device=self.device)
        G1 = torch.matmul(Mass_val_t, term1)
        Gx = torch.cat([G1, term2], dim=0)
        self.Gx = Gx

# ---- CCM-compatible f_func and B_func ----

num_dim_x = 12
num_dim_control = 4

def f_func(x, mp=mp, mq=mq, device=None):
    # x: bs x 12 x 1
    bs = x.shape[0]
    f = torch.zeros(bs, num_dim_x, 1, device=x.device, dtype=x.dtype)
    for i in range(bs):
        fxgx = Fx_Gx_Calculations(x[i,:,0], mp, mq, device=device)
        f[i,:,0] = fxgx.Fx.detach().view(-1)
    return f

def DfDx_func(x, mp=mp, mq=mq, device=None):
    # Returns: bs x n x n
    bs = x.shape[0]
    J = torch.zeros(bs, num_dim_x, num_dim_x, device=x.device, dtype=x.dtype)
    for i in range(bs):
        xi = x[i,:,0].clone().detach().requires_grad_(True)
        def fx_wrap(xi_):
            fxgx = Fx_Gx_Calculations(xi_, mp, mq, device=device)
            return fxgx.Fx
        jac = torch.autograd.functional.jacobian(fx_wrap, xi)
        J[i,:,:] = jac
    return J

def B_func(x, mp=mp, mq=mq, device=None):
    # x: bs x 12 x 1
    bs = x.shape[0]
    B = torch.zeros(bs, num_dim_x, num_dim_control, device=x.device, dtype=x.dtype)
    for i in range(bs):
        fxgx = Fx_Gx_Calculations(x[i,:,0], mp, mq, device=device)
        B[i,:,:] = fxgx.Gx.detach()
    return B

def DBDx_func(x, mp=mp, mq=mq, device=None):
    # Returns: bs x n x n x m
    bs = x.shape[0]
    DBDx = torch.zeros(bs, num_dim_x, num_dim_x, num_dim_control, device=x.device, dtype=x.dtype)
    for i in range(bs):
        xi = x[i,:,0].clone().detach().requires_grad_(True)
        def gx_wrap(xi_):
            fxgx = Fx_Gx_Calculations(xi_, mp, mq, device=device)
            return fxgx.Gx
        jac = torch.autograd.functional.jacobian(gx_wrap, xi)  # shape: (n, n, m)
        DBDx[i,:,:,:] = jac
    return DBDx

# Example usage:
# x = torch.randn(batch_size, 12, 1)
# mp = 1.0
# mq = 1.0
# f = f_func(x, mp, mq)
# B = B_func(x, mp, mq)