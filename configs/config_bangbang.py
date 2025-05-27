import numpy as np

X_MIN = np.array([-5., -2.]).reshape(-1,1)
X_MAX = np.array([5., 2.]).reshape(-1,1)
XE_MIN = np.array([-1., -1.]).reshape(-1,1)
XE_MAX = np.array([1., 1.]).reshape(-1,1)
UREF_MIN = np.array([-0.5]).reshape(-1,1)
UREF_MAX = np.array([0.5]).reshape(-1,1)

time_bound = 5.
time_step = 0.05
t = np.arange(0, time_bound, time_step)

def system_reset(seed):
    np.random.seed(int(seed*1e6)%2**32)
    xref_0 = np.array([[2.],[0.]])
    xe_0 = np.zeros((2,1))
    x_0 = xref_0 + xe_0
    uref = [np.zeros((1,1)) for _ in t]
    return x_0, xref_0, uref