from math import sqrt

from scipy.stats import norm
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

sns.set(style='ticks')


def brownian(x0, n, dt, delta):
    """One dimensional Brownian motion (i.e. the Wiener process)."""
    x0 = np.asarray(x0)
    r = norm.rvs(size=x0.shape + (n,), scale=delta*sqrt(dt))
    brown = np.cumsum(r, axis=-1)
    brown += np.expand_dims(x0, axis=-1)
    return brown


# plot several realizations of 1D Brownian motion
# the Wiener process parameter
delta = 2
# total time
T = 10.0
# number of steps
N = 500
# time step size
dt = T/N
# number of realizations to generate
m = 20
# initial values
x0 = 10 * np.ones(m)

x = brownian(x0, N, dt, delta)
plt.figure()
t = np.linspace(0, N*dt, N)
for k in range(m):
    plt.plot(t, x[k])

sns.despine()
plt.xlabel('Time', fontsize=15)
plt.ylabel('X', fontsize=15)
plt.tight_layout()
plt.savefig('brownian-1d.pdf')


# plot one realization of 2D Brownian motion
delta = 0.25
T = 10.0
N = 500
dt = T/N
x0 = np.zeros(2)

x = brownian(x0, N, dt, delta)
plt.figure()
plt.plot(x[0], x[1], 'k')
plt.plot(x[0,0],x[1,0], 'go', markersize=15)
plt.plot(x[0,-1], x[1,-1], 'ro', markersize=15)

plt.grid('on')
sns.despine()
plt.xlabel('X', fontsize=15)
plt.ylabel('Y', fontsize=15)
plt.tight_layout()
plt.savefig('brownian-2d.pdf')
