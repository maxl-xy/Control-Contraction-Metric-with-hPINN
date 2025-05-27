import torch
from torch.autograd import grad
import torch.nn.functional as F
import matplotlib.pyplot as plt
import importlib
import numpy as np
import time
from tqdm import tqdm

import os
import sys
sys.path.append('systems')
sys.path.append('configs')
sys.path.append('models')
import argparse

np.random.seed(1024)

# Hyperparameters
parser = argparse.ArgumentParser(description="")
parser.add_argument('--task', type=str,
                        default='CAR', help='Name of the model.')
parser.add_argument('--no_cuda', dest='use_cuda', action='store_false', help='Disable cuda.')
parser.set_defaults(use_cuda=True)
parser.add_argument('--bs', type=int, default=1024, help='Batch size.')
parser.add_argument('--num_train', type=int, default=131072, help='Number of samples for training.') # 4096 * 32
parser.add_argument('--num_test', type=int, default=32768, help='Number of samples for testing.') # 1024 * 32
parser.add_argument('--lr', dest='learning_rate', type=float, default=0.001, help='Base learning rate.')
parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs.')
parser.add_argument('--lr_step', type=int, default=5, help='')
parser.add_argument('--lambda', type=float, dest='_lambda', default=0.5, help='Convergence rate: lambda')
parser.add_argument('--w_ub', type=float, default=10, help='Upper bound of the eigenvalue of the dual metric.')
parser.add_argument('--w_lb', type=float, default=0.1, help='Lower bound of the eigenvalue of the dual metric.')
parser.add_argument('--log', type=str, help='Path to a directory for storing the log.')

args = parser.parse_args()

os.system('cp *.py '+args.log)
os.system('cp -r models/ '+args.log)
os.system('cp -r configs/ '+args.log)
os.system('cp -r systems/ '+args.log)

epsilon = args._lambda * 0.1

config = importlib.import_module('config_'+args.task)
X_MIN = config.X_MIN
X_MAX = config.X_MAX
U_MIN = config.UREF_MIN
U_MAX = config.UREF_MAX
XE_MIN = config.XE_MIN
XE_MAX = config.XE_MAX

system = importlib.import_module('system_'+args.task)
f_func = system.f_func
B_func = system.B_func
num_dim_x = system.num_dim_x
num_dim_control = system.num_dim_control
if hasattr(system, 'Bbot_func'):
    Bbot_func = system.Bbot_func

model = importlib.import_module('model_'+args.task)
get_model = model.get_model

model_W, model_Wbot, model_u_w1, model_u_w2, W_func, u_func = get_model(num_dim_x, num_dim_control, w_lb=args.w_lb, use_cuda=args.use_cuda)

# constructing datasets
def sample_xef():
    return (X_MAX-X_MIN) * np.random.rand(num_dim_x, 1) + X_MIN

def sample_x(xref):
    xe = (XE_MAX-XE_MIN) * np.random.rand(num_dim_x, 1) + XE_MIN
    x = xref + xe
    x[x>X_MAX] = X_MAX[x>X_MAX]
    x[x<X_MIN] = X_MIN[x<X_MIN]
    return x

def sample_uref():
    return (U_MAX-U_MIN) * np.random.rand(num_dim_control, 1) + U_MIN

def sample_full():
    xref = sample_xef()
    uref = sample_uref()
    x = sample_x(xref)
    return (x, xref, uref)

X_tr = [sample_full() for _ in range(args.num_train)]
X_te = [sample_full() for _ in range(args.num_test)]

if 'Bbot_func' not in locals():
    def Bbot_func(x): # columns of Bbot forms a basis of the null space of B^T
        bs = x.shape[0]
        Bbot = torch.cat((torch.eye(num_dim_x-num_dim_control, num_dim_x-num_dim_control),
            torch.zeros(num_dim_control, num_dim_x-num_dim_control)), dim=0)
        if args.use_cuda:
            Bbot = Bbot.cuda()
        Bbot.unsqueeze(0)
        return Bbot.repeat(bs, 1, 1)

def Jacobian_Matrix(M, x):
    # NOTE that this function assume that data are independent of each other
    # along the batch dimension.
    # M: B x m x m
    # x: B x n x 1
    # ret: B x m x m x n
    bs = x.shape[0]
    m = M.size(-1)
    n = x.size(1)
    J = torch.zeros(bs, m, m, n).type(x.type())
    for i in range(m):
        for j in range(m):
            J[:, i, j, :] = grad(M[:, i, j].sum(), x, create_graph=True)[0].squeeze(-1)
    return J

def Jacobian(f, x):
    # NOTE that this function assume that data are independent of each other
    f = f + 0. * x.sum() # to avoid the case that f is independent of x
    # f: B x m x 1
    # x: B x n x 1
    # ret: B x m x n
    bs = x.shape[0]
    m = f.size(1)
    n = x.size(1)
    J = torch.zeros(bs, m, n).type(x.type())
    for i in range(m):
        J[:, i, :] = grad(f[:, i, 0].sum(), x, create_graph=True)[0].squeeze(-1)
    return J

def weighted_gradients(W, v, x, detach=False):
    # v, x: bs x n x 1
    # DWDx: bs x n x n x n
    assert v.size() == x.size()
    bs = x.shape[0]
    if detach:
        return (Jacobian_Matrix(W, x).detach() * v.view(bs, 1, 1, -1)).sum(dim=3)
    else:
        return (Jacobian_Matrix(W, x) * v.view(bs, 1, 1, -1)).sum(dim=3)

K = 1024
def loss_pos_matrix_random_sampling(A):
    # A: bs x d x d
    # z: K x d
    if args.use_cuda:
        z = torch.randn(K, A.size(-1)).cuda()
    else:
        z = torch.randn(K, A.size(-1))
    z = z / z.norm(dim=1, keepdim=True)
    zTAz = (z.matmul(A) * z.view(1,K,-1)).sum(dim=2).view(-1)
    negative_index = zTAz.detach().cpu().numpy() < 0
    if negative_index.sum()>0:
        negative_zTAz = zTAz[negative_index]
        return -1.0 * (negative_zTAz.mean())
    else:
        return torch.tensor(0.).type(z.type()).requires_grad_()

def loss_pos_matrix_eigen_values(A):
    # A: bs x d x d
    eigv = torch.linalg.eigh(A, UPLO='U')[0].view(-1)
    negative_index = eigv.detach().cpu().numpy() < 0
    negative_eigv = eigv[negative_index]
    return negative_eigv.norm()

def forward(x, xref, uref, _lambda, verbose=False, acc=False, detach=False):
    # x: bs x n x 1
    bs = x.shape[0]
    W = W_func(x)
    M = torch.inverse(W)
    f = f_func(x)
    B = B_func(x)
    DfDx = Jacobian(f, x)
    DBDx = torch.zeros(bs, num_dim_x, num_dim_x, num_dim_control).type(x.type())
    for i in range(num_dim_control):
        DBDx[:,:,:,i] = Jacobian(B[:,:,i].unsqueeze(-1), x)

    _Bbot = Bbot_func(x)
    u = u_func(x, x - xref, uref) # u: bs x m x 1 # TODO: x - xref
    K = Jacobian(u, x)

    A = DfDx + sum([u[:, i, 0].unsqueeze(-1).unsqueeze(-1) * DBDx[:, :, :, i] for i in range(num_dim_control)])
    dot_x = f + B.matmul(u)
    dot_M = weighted_gradients(M, dot_x, x, detach=detach) # DMDt
    dot_W = weighted_gradients(W, dot_x, x, detach=detach) # DWDt
    if detach:
        Contraction = dot_M + (A + B.matmul(K)).transpose(1,2).matmul(M.detach()) + M.detach().matmul(A + B.matmul(K)) + 2 * _lambda * M.detach()
    else:
        Contraction = dot_M + (A + B.matmul(K)).transpose(1,2).matmul(M) + M.matmul(A + B.matmul(K)) + 2 * _lambda * M

    # C1
    C1_inner = - weighted_gradients(W, f, x) + DfDx.matmul(W) + W.matmul(DfDx.transpose(1,2)) + 2 * _lambda * W
    C1_LHS_1 = _Bbot.transpose(1,2).matmul(C1_inner).matmul(_Bbot) # this has to be a negative definite matrix

    # C2
    C2_inners = []
    C2s = []
    for j in range(num_dim_control):
        C2_inner = weighted_gradients(W, B[:,:,j].unsqueeze(-1), x) - (DBDx[:,:,:,j].matmul(W) + W.matmul(DBDx[:,:,:,j].transpose(1,2)))
        C2 = _Bbot.transpose(1,2).matmul(C2_inner).matmul(_Bbot)
        C2_inners.append(C2_inner)
        C2s.append(C2)

    loss = 0
    loss += loss_pos_matrix_random_sampling(-Contraction - epsilon * torch.eye(Contraction.shape[-1]).unsqueeze(0).type(x.type()))
    loss += loss_pos_matrix_random_sampling(-C1_LHS_1 - epsilon * torch.eye(C1_LHS_1.shape[-1]).unsqueeze(0).type(x.type()))
    loss += loss_pos_matrix_random_sampling(args.w_ub * torch.eye(W.shape[-1]).unsqueeze(0).type(x.type()) - W)
    loss += 1. * sum([1.*(C2**2).reshape(bs,-1).sum(dim=1).mean() for C2 in C2s])

    if verbose:
        eigenvalues = torch.linalg.eigh(Contraction, UPLO='U')[0]
        print(eigenvalues.min(dim=1)[0].mean(), eigenvalues.max(dim=1)[0].mean(), eigenvalues.mean())

    if acc:
        return loss, \
    ((torch.linalg.eigh(Contraction, UPLO='U')[0] >= 0).sum(dim=1) == 0).cpu().detach().numpy(), \
    ((torch.linalg.eigh(C1_LHS_1, UPLO='U')[0] >= 0).sum(dim=1) == 0).cpu().detach().numpy(), \
    sum([1. * (C2**2).reshape(bs, -1).sum(dim=1).mean() for C2 in C2s]).item()
    else:
        return loss, None, None, None

# For computing diagnostics
def compute_diagnostics(x, xref, uref, _lambda):
    """
    Returns three np.arrays of shape (bs,):
      - max_eig_C1:    the largest eigenvalue of C1 (should be < 0)
      - max_norm_C2:   the largest Frobenius‐norm of any C2_j (should be ~ 0)
      - max_eig_Contr: the largest eigenvalue of the contraction matrix (Eq.4, should be < 0)
    """
    # pull in everything
    W = W_func(x)                    # bs×n×n
    M = torch.inverse(W)             # bs×n×n
    f = f_func(x)                    # bs×n×1
    B = B_func(x)                    # bs×n×m

    # Jacobians
    DfDx = Jacobian(f, x)            # bs×n×n
    DBDx = torch.stack([
        Jacobian(B[:,:,j].unsqueeze(-1), x)
        for j in range(num_dim_control)
    ], dim=-1)                       # bs×n×n×m

    Bbot = Bbot_func(x)              # bs×n×(n−m)

    # --- Condition 2 (C1) ---
    C1_inner = (
        -weighted_gradients(W, f, x)
        + DfDx @ W
        + W @ DfDx.transpose(1,2)
        + 2*_lambda * W
    )                                 # bs×n×n
    C1 = Bbot.transpose(1,2) @ C1_inner @ Bbot
    eigs_C1 = torch.linalg.eigvalsh(C1)      # bs×(n−m)
    max_eig_C1 = eigs_C1.max(dim=1)[0]       # bs

    # --- Condition 3 (C2) ---
    norms_C2 = []
    for j in range(num_dim_control):
        C2_inner = (
            weighted_gradients(W, B[:,:,j].unsqueeze(-1), x)
            - (DBDx[:,:,: ,j] @ W + W @ DBDx[:,:,: ,j].transpose(1,2))
        )
        C2 = Bbot.transpose(1,2) @ C2_inner @ Bbot
        norms_C2.append(torch.norm(C2.reshape(C2.shape[0], -1), dim=1))
    max_norm_C2 = torch.stack(norms_C2, dim=1).max(dim=1)[0]  # bs

    # --- Condition 4 (Contraction on M) ---
    # 1) u(x)
    u = u_func(x, x - xref, uref)           # bs×m×1
    dot_x = f + B.matmul(u)                 # bs×n×1

    # 2) dot_M = ∂ₓM · dot_x
    dot_M = weighted_gradients(M, dot_x, x)

    # 3) A = DfDx + Σ u_j ∂ₓb_j
    A = DfDx.clone()
    for j in range(num_dim_control):
        A = A + u[:,j,0].view(-1,1,1) * DBDx[:,:,:,j]

    # 4) K = ∂ₓu, so BK = B·K
    K = Jacobian(u, x)                     # bs×m×n
    BK = B.matmul(K)                       # bs×n×n

    # 5) build contraction matrix
    Contr = (
        dot_M
        + (A + BK).transpose(1,2) @ M
        + M @ (A + BK)
        + 2*_lambda * M
    )                                       # bs×n×n
    eigs_Contr = torch.linalg.eigvalsh(Contr)  # bs×n
    max_eig_Contr = eigs_Contr.max(dim=1)[0]   # bs

    # move everything to cpu & numpy
    return (
        max_eig_C1.detach().cpu().numpy(),
        max_norm_C2.detach().cpu().numpy(),
        max_eig_Contr.detach().cpu().numpy()
    )


optimizer = torch.optim.Adam(list(model_W.parameters()) + list(model_Wbot.parameters()) + list(model_u_w1.parameters()) + list(model_u_w2.parameters()), lr=args.learning_rate)

def trainval(X, bs=args.bs, train=True, _lambda=args._lambda, acc=False, detach=False): # trainval a set of x
    # torch.autograd.set_detect_anomaly(True)

    if train:
        indices = np.random.permutation(len(X))
    else:
        indices = np.array(list(range(len(X))))

    total_loss = 0
    total_p1 = 0
    total_p2 = 0
    total_l3 = 0

    if train:
        _iter = tqdm(range(len(X) // bs))
    else:
        _iter = range(len(X) // bs)
    for b in _iter:
        start = time.time()
        x = []; xref = []; uref = [];
        for id in indices[b*bs:(b+1)*bs]:
            if args.use_cuda:
                x.append(torch.from_numpy(X[id][0]).float().cuda())
                xref.append(torch.from_numpy(X[id][1]).float().cuda())
                uref.append(torch.from_numpy(X[id][2]).float().cuda())
            else:
                x.append(torch.from_numpy(X[id][0]).float())
                xref.append(torch.from_numpy(X[id][1]).float())
                uref.append(torch.from_numpy(X[id][2]).float())

        x, xref, uref = (torch.stack(d).detach() for d in (x, xref, uref))
        x = x.requires_grad_()

        start = time.time()

        loss, p1, p2, l3 = forward(x, xref, uref, _lambda=_lambda, verbose=False if not train else False, acc=acc, detach=detach)

        start = time.time()
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print('backwad(): %.3f s'%(time.time() - start))

        total_loss += loss.item() * x.shape[0]
        if acc:
            total_p1 += p1.sum()
            total_p2 += p2.sum()
            total_l3 += l3 * x.shape[0]
    return total_loss / len(X), total_p1 / len(X), total_p2 / len(X), total_l3/ len(X)


best_acc = 0

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by every args.lr_step epochs"""
    lr = args.learning_rate * (0.1 ** (epoch // args.lr_step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# just before the epoch loop, define histories
history_C1    = []
history_C2    = []
history_Contr = []

for epoch in range(args.epochs):
    adjust_learning_rate(optimizer, epoch)
    # --- training & basic test as before ---
    loss, _, _, _ = trainval(X_tr, train=True,
                             _lambda=args._lambda,
                             acc=False, detach=(epoch<args.lr_step))
    print("Training loss: ", loss)
    loss, p1, p2, l3 = trainval(X_te, train=False,
                                _lambda=0., acc=True, detach=False)
    print(f"Epoch {epoch}: Testing loss/p1/p2/l3:", loss, p1, p2, l3)

    # --- new diagnostics collection for Eq(2), Eq(3), and Eq(4) ---
    all_C1, all_C2, all_Contr = [], [], []
    for b in range(len(X_te)//args.bs):
        batch = X_te[b*args.bs:(b+1)*args.bs]
        x_batch   = torch.stack([torch.from_numpy(t[0]).float() for t in batch]).requires_grad_(True)
        xref_batch= torch.stack([torch.from_numpy(t[1]).float() for t in batch])
        uref_batch= torch.stack([torch.from_numpy(t[2]).float() for t in batch])
        if args.use_cuda:
            x_batch, xref_batch, uref_batch = x_batch.cuda(), xref_batch.cuda(), uref_batch.cuda()

        c1_vals, c2_vals, contr_vals = compute_diagnostics(x_batch, xref_batch, uref_batch, _lambda=args._lambda)
        all_C1.append(c1_vals)
        all_C2.append(c2_vals)
        all_Contr.append(contr_vals)

    all_C1    = np.concatenate(all_C1)
    all_C2    = np.concatenate(all_C2)
    all_Contr = np.concatenate(all_Contr)

    # record worst‐cases
    history_C1.append(all_C1.min())     # smallest max‐eig(C1)
    history_C2.append(all_C2.max())     # largest norm(C2)
    history_Contr.append(all_Contr.min())# smallest max‐eig(Contraction)

    if p1+p2 >= best_acc:
        best_acc = p1 + p2
        filename = args.log+'/model_best.pth.tar'
        filename_controller = args.log+'/controller_best.pth.tar'
        torch.save({'args':args, 'precs':(loss, p1, p2), 'model_W': model_W.state_dict(), 'model_Wbot': model_Wbot.state_dict(), 'model_u_w1': model_u_w1.state_dict(), 'model_u_w2': model_u_w2.state_dict()}, filename)
        #torch.save(u_func, filename_controller)


# --- plot the diagnostics ---
epochs = np.arange(len(history_C1))

plt.figure()
plt.plot(epochs, history_C1, marker='o')
plt.axhline(0, linestyle='--', color='k')
plt.title(r'Worst‐case $\max\lambda(C_1)$ per Epoch (Eq. 2)')
plt.xlabel('Epoch'); plt.ylabel('Max Eig(C₁)'); plt.grid(True)

plt.figure()
plt.plot(epochs, history_C2, marker='o')
plt.axhline(0, linestyle='--', color='k')
plt.title(r'Worst‐case $\max_j\|C_{2,j}\|$ per Epoch (Eq. 3)')
plt.xlabel('Epoch'); plt.ylabel('Max ‖C₂‖'); plt.grid(True)

plt.figure()
plt.plot(epochs, history_Contr, marker='o')
plt.axhline(0, linestyle='--', color='k')
plt.title(r'Worst‐case $\max\lambda\!\left(\dot M + M(A+BK) + 2\lambda M\right)$ per Epoch (Eq. 4)')
plt.xlabel('Epoch'); plt.ylabel('Max Eig(Contraction)'); plt.grid(True)

plt.show()