import numpy as np
from utils import temp_seed

# for ours training
X_MIN = np.array([-15., -15., -15., -2, -2, -2, -np.pi/3, -np.pi/3, -np.pi/3, -np.pi/3, -np.pi/3, -np.pi/3]).reshape(-1,1)
X_MAX = np.array([ 15.,  15.,  15.,  2,  2,  2,  np.pi/3,  np.pi/3, np.pi/3, np.pi/3,  np.pi/3, np.pi/3]).reshape(-1,1)

g = 9.81

tau_x_bound = np.pi / 3
tau_y_bound = np.pi / 3
tau_z_bound = np.pi / 3 
f_bound_u = 20. 
f_bound_l = -5. 

UREF_MIN = np.array([f_bound_u, -tau_x_bound, -tau_y_bound, -tau_z_bound]).reshape(-1,1)
UREF_MAX = np.array([f_bound_l,  tau_x_bound, tau_y_bound, tau_z_bound]).reshape(-1,1)

lim = 1.
XE_MIN = np.array([-lim, -lim, -lim, -lim, -lim, -lim, -lim, -lim, -lim, -lim, -lim, -lim]).reshape(-1,1)
XE_MAX = np.array([lim, lim, lim, lim, lim, lim, lim, lim, lim, lim, lim, lim]).reshape(-1,1)

# for sampling ref
X_INIT_MIN = np.array([-2., -2, -2, -1, -1, -1, 0, 0, 0, 0, 0, 0])
X_INIT_MAX = np.array([ 2.,  2,  2,  1,  1, 1, 0, 0, 0, 0, 0, 0])


XE_INIT_MIN = np.array([-0.5,]*12)
XE_INIT_MAX = np.array([ 0.5,]*12)

time_bound = 6.
time_step = 0.03
t = np.arange(0, time_bound, time_step)

def system_reset(seed):
    SEED_MAX = 10000000
    with temp_seed(int(seed * SEED_MAX)):
        xref_0 = X_INIT_MIN + np.random.rand(len(X_INIT_MIN)) * (X_INIT_MAX - X_INIT_MIN)
        xe_0 = XE_INIT_MIN + np.random.rand(len(XE_INIT_MIN)) * (XE_INIT_MAX - XE_INIT_MIN)
        x_0 = xref_0 + xe_0

        g = 9.81
        freqs = list(range(1,11))
        # freqs = []
        weights = np.random.randn(len(freqs), len(UREF_MIN))
        weights = (2. * weights / np.sqrt((weights**2).sum(axis=0, keepdims=True))).tolist()
        uref = []
        for _t in t:
            u = np.array([0., 0., 0., 0.]) # ref
            for freq, weight in zip(freqs, weights):
                u += np.array([weight[0] * np.sin(freq * _t/time_bound * 2*np.pi), 0.1*weight[1] * np.sin(freq * _t/time_bound * 2*np.pi), 0.1*weight[2] * np.sin(freq * _t/time_bound * 2*np.pi), 0.1*weight[2] * np.sin(freq * _t/time_bound * 2*np.pi)])
            # u += 0.01*np.random.randn(2)
            uref.append(u)

    return x_0, xref_0, uref
