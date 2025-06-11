import numpy as np

# ------------------ Physical Constants ------------------
g = 9.81

# Masses
MP = 0.5        # Mass of the payload
MQ = 1.63       # Mass of the quadcopter

# Cable length (nominal)
L0 = 1.0        # Nominal cable length

# ------------------ State Indices -----------------------
# [0:2]   r      (payload position, 2D)
# [2:4]   v      (payload velocity, 2D)
# [4:7]   posq   (quadrotor position, 3D)
# [7:10]  vq     (quadrotor velocity, 3D)
# [10]    l      (cable length, 1D)
# [11]    l_dot  (cable length rate, 1D)

# ------------------ State Limits ------------------------
X_MIN = np.array([
    -2.0,  -2.0,      # r
    -2.0,  -2.0,      # v
    -10.0, -10.0, -10.0,  # posq
    -5.0,  -5.0,  -5.0,   # vq
    0.5,              # l (cable length, must be positive)
    -2.0              # l_dot
]).reshape(-1,1)

X_MAX = np.array([
     2.0,   2.0,      # r
     2.0,   2.0,      # v
    10.0,  10.0,  10.0,   # posq
     5.0,   5.0,   5.0,   # vq
     2.0,              # l
     2.0               # l_dot
]).reshape(-1,1)

# ------------------ Control Limits ----------------------
UREF_MIN = np.array([-10., -10., -10., -10.]).reshape(-1,1)
UREF_MAX = np.array([ 10.,  10.,  10.,  10.]).reshape(-1,1)

# ------------------ Error State Limits ------------------
lim = 1.
XE_MIN = np.array([-lim]*12).reshape(-1,1)
XE_MAX = np.array([ lim]*12).reshape(-1,1)

# ------------------ Initial State Sampling --------------
X_INIT_MIN = np.array([
    -1.0, -1.0,    # r
    -0.5, -0.5,    # v
    -2.0, -2.0, -2.0,  # posq
    -1.0, -1.0, -1.0,  # vq
    1.0,           # l
    0.0            # l_dot
])
X_INIT_MAX = np.array([
     1.0,  1.0,    # r
     0.5,  0.5,    # v
     2.0,  2.0,  2.0,   # posq
     1.0,  1.0,  1.0,   # vq
     1.5,           # l
     0.0            # l_dot
])

XE_INIT_MIN = np.array([-0.5]*12)
XE_INIT_MAX = np.array([ 0.5]*12)

# ------------------ Time Settings -----------------------
time_bound = 6.
time_step = 0.03
t = np.arange(0, time_bound, time_step)

# ------------------ Learning/Simulation Params ----------
# For compatibility with original main
dt = 1.0
t_span = (0, 0.3)
samples_req = int((t_span[1] - t_span[0]) / dt + 1)
t_eval = np.linspace(*t_span, samples_req)
alims = np.array([0.1, 1.0])
elims = np.array([3.0, 5.0])
xlims = np.array([
    [-0.1, -0.1, -0.5, -0.5, -1, -1, -1, -0.7, -0.7, -0.7, 1, -0.5],
    [ 0.1,  0.1,  0.5,  0.5,  1,  1,  1,  0.7,  0.7,  0.7, 1.2, 0.5]
])
Nx = 12
Nls = 10
fname = "test_data"

# ------------------ State Weights -----------------------
state_weights = np.ones(12)

# ------------------ System Reset Function ---------------
from utils import temp_seed

def system_reset(seed):
    SEED_MAX = 10000000
    with temp_seed(int(seed * SEED_MAX)):
        xref_0 = X_INIT_MIN + np.random.rand(len(X_INIT_MIN)) * (X_INIT_MAX - X_INIT_MIN)
        xe_0 = XE_INIT_MIN + np.random.rand(len(XE_INIT_MIN)) * (XE_INIT_MAX - XE_INIT_MIN)
        x_0 = xref_0 + xe_0

        freqs = list(range(1, 12+1))
        weights = np.random.randn(len(freqs), len(UREF_MIN))
        weights = (0.1 * weights / np.sqrt((weights**2).sum(axis=0, keepdims=True))).tolist()
        uref = []
        for _t in t:
            u = np.zeros(4) # 4D control input
            for freq, weight in zip(freqs, weights):
                u += np.array([
                    weight[0] * np.sin(freq * _t/time_bound * 2*np.pi),
                    weight[1] * np.sin(freq * _t/time_bound * 2*np.pi),
                    weight[2] * np.sin(freq * _t/time_bound * 2*np.pi),
                    weight[3] * np.sin(freq * _t/time_bound * 2*np.pi),
                ])
            uref.append(u)

    return x_0, xref_0, uref