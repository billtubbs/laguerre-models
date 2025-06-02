import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from laguerre_model import make_step_response_function


def make_step_response_plot(t, X, Y):

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7, 4))

    N = X.shape[1]
    x_labels = [f"$x_{i+1}(t)$" for i in range(N)]
    y_label = "$y(t)$"

    ax = axes[0]
    ax.plot(t, X, marker='.', linestyle='-', label=x_labels)
    ax.grid()
    ax.legend()

    ax = axes[1]
    ax.plot(t, Y, marker='.', linestyle='-', label=y_label)
    ax.set_xlabel('t')
    ax.grid()
    ax.legend()

    return fig, ax

# Example
N = 4
p = 0.5
c = np.ones(N)

# Simulation time
nT = 20
T = 0.5
t = ca.DM(T * np.arange(nT+1))

step_response = make_step_response_function(T, nT)

t = T * np.arange(nT+1)

X, Y = step_response(p, c)
X = np.array(X)
Y = np.array(Y)

fig, ax = make_step_response_plot(t, X, Y)
plt.tight_layout()
plt.show()