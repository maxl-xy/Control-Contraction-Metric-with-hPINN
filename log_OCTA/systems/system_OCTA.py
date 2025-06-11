import torch

num_dim_x = 12
num_dim_control = 4

g = 9.81
m = 10.0  # mass of the octocopter
I_xx = 0.2  # moment of inertia around x-axis
I_yy = 0.2  # moment of inertia around y-axis
I_zz = 0.2  # moment of inertia around z-axis

def f_func(x):
    # x: bs x n x 1
    # f: bs x n x 1
    bs = x.shape[0]
    x, y, z, u, v, w, phi, theta, psi, p, q, r = [x[:,i,0] for i in range(num_dim_x)]

    # Compute rotation matrix for each batch
    c_phi = torch.cos(phi)
    s_phi = torch.sin(phi)
    c_theta = torch.cos(theta)
    s_theta = torch.sin(theta)
    c_psi = torch.cos(psi)
    s_psi = torch.sin(psi)

    # Each element is (bs,)
    R11 = c_psi * c_theta
    R12 = c_psi * s_theta * s_phi - s_psi * c_phi
    R13 = c_psi * s_theta * c_phi + s_psi * s_phi
    R21 = s_psi * c_theta
    R22 = s_psi * s_theta * s_phi + c_psi * c_phi
    R23 = s_psi * s_theta * c_phi - c_psi * s_phi
    R31 = -s_theta
    R32 = c_theta * s_phi
    R33 = c_theta * c_phi

    # Stack into rotation matrix: (bs, 3, 3)
    R_EB = torch.stack([
        torch.stack([R11, R12, R13], dim=1),
        torch.stack([R21, R22, R23], dim=1),
        torch.stack([R31, R32, R33], dim=1)
    ], dim=1)

    I = torch.tensor([I_xx, I_yy, I_zz], device=x.device)  # moment of inertia vector
    I_B = torch.diag_embed(I)  # moment of inertia matrix in body frame
    
    # Compute velocities in body frame
    # Stack velocities: (bs, 3, 1)
    vel = torch.stack([u, v, w], dim=1).unsqueeze(-1)  # (bs, 3, 1)

    # Apply rotation: (bs, 3, 1)
    pos_dot = torch.bmm(R_EB, vel)
    
    f = torch.zeros(bs, num_dim_x, 1).type(x.type())
    f[:, 0:3, 0] = pos_dot.squeeze(-1)
    
    # Compute linear acceleration in body frame
    ang_vel = torch.stack([p, q, r], dim=1).unsqueeze(-1)  # (bs, 3, 1)
    accel = torch.cross(vel.squeeze(-1), ang_vel.squeeze(-1), dim=1)  # (bs, 3)
    gravity = torch.tensor([0, 0, -g], device=x.device).view(1, 3, 1).expand(bs, 3, 1)  # (bs, 3, 1)
    R_BE = R_EB.transpose(1, 2)  # (bs, 3, 3)
    f[:, 3:6, 0] = accel + torch.bmm(R_BE, gravity).squeeze(-1)
    
    # Compute angular velocity in body frame
    S_B_inv = torch.stack([
        torch.stack([
            torch.ones_like(phi),
            torch.sin(phi) * torch.tan(theta),
            torch.cos(phi) * torch.tan(theta)
        ], dim=1),
        torch.stack([
            torch.zeros_like(phi),
            torch.cos(phi),
            -torch.sin(phi)
        ], dim=1),
        torch.stack([
            torch.zeros_like(phi),
            torch.sin(phi) / torch.cos(theta),
            torch.cos(phi) / torch.cos(theta)
        ], dim=1)
    ], dim=1)  # (bs, 3, 3)
    f[:, 6:9, 0] = torch.bmm(S_B_inv, ang_vel).squeeze(-1)
    
    # Compute angular acceleration in body frame
    I_B_batch = I_B.unsqueeze(0).expand(bs, -1, -1)  # (bs, 3, 3)
    result = torch.bmm(I_B_batch, ang_vel.squeeze(-1).unsqueeze(-1)).squeeze(-1)  # shape: (bs, 3)
    f[:, 9:12, 0] = -torch.bmm(
        torch.linalg.inv(I_B_batch),
        torch.cross(
            ang_vel.squeeze(-1),
            result,
            dim=1
        ).unsqueeze(-1)
    ).squeeze(-1)
    
    return f

def DfDx_func(x):
    raise NotImplemented('NotImplemented')

def B_func(x):
    bs = x.shape[0]
    
    I = torch.tensor([I_xx, I_yy, I_zz], device=x.device)  # moment of inertia vector
    I_B = torch.diag_embed(I)  # moment of inertia matrix in body frame
    
    B = torch.zeros(bs, num_dim_x, num_dim_control).type(x.type())

    B[:, 5, 0] = 1/m  # Thrust force in z direction of body-fixed frame
    B[:, 9:12, 1:4] = I_B.inverse()  # Moments of inertia for roll, pitch, yaw
    return B

def DBDx_func(x):
    raise NotImplemented('NotImplemented')
