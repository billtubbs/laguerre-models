import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def calculate_tau_parameters(p, T):
    """
    Calculate the tau parameters from the Laguerre filter theory.

    Returns:
    --------
    Tuple of CasADi symbolic expressions for tau1, tau2, tau3, tau4
    """

    # Calculate tau parameters as per equations in the paper
    exp_pT = ca.exp(-p * T)

    tau1 = exp_pT
    tau2 = T + (2 / p) * (exp_pT - 1)
    tau3 = -T * exp_pT - (2 / p) * (exp_pT - 1)
    tau4 = ca.sqrt(2 * p * (1 - tau1) / p)

    return tau1, tau2, tau3, tau4


def construct_A_matrix(N, T, tau1, tau2, tau3) -> ca.SX:
    """
    Construct the A matrix based on the Laguerre filter structure.

    Returns:
    --------
    ca.SX: N x N state transition matrix
    """
    A = ca.SX.zeros(N, N)

    # Fill the diagonal with tau1
    for i in range(N):
        A[i, i] = tau1

    # Fill the super-diagonal elements
    for i in range(N - 1):
        if i == 0:
            # First row, second column
            A[i, i + 1] = 0
        else:
            # Other super-diagonal elements
            A[i, i + 1] = 0

    # Fill the sub-diagonal elements
    for i in range(1, N):
        A[i, i - 1] = -(tau1 * tau2 + tau3) / T

    # Fill the first column (except diagonal)
    for i in range(1, N):
        A[i, 0] = ((-1)**(i) * tau1**(i-1) * (tau1 * tau2 + tau3)) / (T**(i))

    return A


def construct_b_vector(N, T, tau2, tau4):
    """
    Construct the b vector (input matrix).

    Returns:
    --------
    ca.SX: N x 1 input vector
    """
    b = ca.SX.zeros(N)

    # Fill b vector according to equation (9)
    for i in range(N):
        if i == 0:
            b[i] = tau4
        else:
            b[i] = (-tau2 / T)**(i) * tau4

    return b


def make_state_transition_function(N):
    p = ca.SX.sym('p')
    T = ca.SX.sym('T')
    tau1, tau2, tau3, tau4 = calculate_tau_parameters(p, T)
    A = construct_A_matrix(N, T, tau1, tau2, tau3)
    assert A.shape == (N, N)
    b = construct_b_vector(N, T, tau2, tau4)
    x = ca.SX.sym('x', N)
    u = ca.SX.sym('u')
    assert b.shape == (N, 1)
    f = ca.Function(
        "f",
        [x, u, p, T],
        [A @ x + b @ u],
        ['xk', 'uk', 'p', 'T'],
        ['xkp1']
    )
    return f


def make_output_function(N):
    x = ca.SX.sym('x', N)
    u = ca.SX.sym('u')
    c = ca.SX.sym('c', N)
    h = ca.Function("h", [x, u, c], [c.T @ x], ['xk', 'uk', 'c'], ['yk'])
    return h


def make_step_response_function(N, T, nT, k_step=2):

    # Construct Laguerre polynomial state transition function
    f = make_state_transition_function(N)
    h = make_output_function(N)

    # Sample period
    T = ca.DM(T)

    # System parameters (variable)
    p = ca.SX.sym('p')
    c = ca.SX.sym('c', N)

    # Unit step input
    U = ca.DM.zeros(nT+1, 1)
    U[k_step:, :] = 1.0

    X = []
    x = ca.DM.zeros(N)
    for k in range(nT+1):
        X.append(x.T)
        u = U[k, :]
        x = f(x, u, p, T)
    X = ca.vcat(X)
    Y = h(X.T, U.T, c).T

    step_response = ca.Function(
        "step_response",
        [p, c],
        [X, Y],
        ['p', 'c'],
        ['X', 'Y']
    )

    return step_response


def make_step_response_plot(t, X, Y, figsize=(7, 4)):

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=figsize)

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


def create_interactive_step_response_plot():
    """
    Creates an interactive step response plot with sliders for parameters
    p and c vector.
    """

    # Fixed parameters
    N = 5
    nT = 50
    T = 0.5
    t = T * np.arange(nT+1)

    # Create the step response function
    step_response = make_step_response_function(N, T, nT)

    # Create figure and subplots
    fig = plt.figure(figsize=(10, 8))

    # Main plot area - leave space at bottom for sliders
    gs = fig.add_gridspec(
        2, 1,
        height_ratios=[1, 1], hspace=0.1,
        left=0.1, right=0.95,
        top=0.85, bottom=0.35
    )

    ax1 = fig.add_subplot(gs[0])  # States plot
    ax2 = fig.add_subplot(gs[1])  # Output plot

    # Initial parameter values
    p_init = 0.5
    c_init = np.zeros(N)

    # Compute initial step response
    X_init, Y_init = step_response(p_init, c_init)
    X_init = np.array(X_init)
    Y_init = np.array(Y_init)

    # Create initial plots
    x_labels = [f"$x_{i+1}(t)$" for i in range(N)]
    y_label = "$y(t)$"

    # Plot states
    lines_x = []
    for i in range(N):
        line, = ax1.plot(
            t,
            X_init[:, i],
            marker='.',
            linestyle='-',
            label=x_labels[i],
        )
        lines_x.append(line)

    ax1.grid(True)
    ax1.legend()
    ax1.set_ylabel('States')
    ax1.set_title('Laguerre Model Step Response')

    # Plot output
    line_y, = ax2.plot(
        t,
        Y_init,
        marker='.',
        linestyle='-',
        label=y_label,
        color='red'
    )
    ax2.grid(True)
    ax2.legend()
    ax2.set_xlabel('Time (t)')
    ax2.set_ylabel('Output')

    # Create slider axes
    slider_height = 0.03
    slider_spacing = 0.04
    slider_left = 0.15
    slider_width = 0.7

    # Parameter p slider
    ax_p = fig.add_axes([slider_left, 0.25, slider_width, slider_height])
    slider_p = Slider(ax_p, 'p', 0.1, 1.0, valinit=p_init, valstep=0.1, valfmt='%.1f')

    # Parameter c sliders
    sliders_c = []
    for i in range(N):
        ax_c = fig.add_axes(
            [
                slider_left,
                0.25 - (i + 1) * slider_spacing,
                slider_width,
                slider_height
            ]
        )
        slider_c = Slider(
            ax_c, f'c[{i+1}]', -2.0, 2.0, valinit=c_init[i], valstep=0.1, valfmt='%.1f'
        )
        sliders_c.append(slider_c)

    def update_plot(val=None):
        """Update the plot when sliders change"""
        # Get current parameter values
        p_current = slider_p.val
        c_current = np.array([slider.val for slider in sliders_c])

        try:
            # Compute new step response
            X_new, Y_new = step_response(p_current, c_current)
            X_new = np.array(X_new)
            Y_new = np.array(Y_new)

            # Update state plots
            for i, line in enumerate(lines_x):
                line.set_ydata(X_new[:, i])

            # Update output plot
            line_y.set_ydata(Y_new)

            # Rescale axes if needed
            ax1.relim()
            ax1.autoscale_view()
            ax2.relim()
            ax2.autoscale_view()

            # Redraw
            fig.canvas.draw_idle()

        except Exception as e:
            print(f"Error updating plot: {e}")

    # Connect sliders to update function
    slider_p.on_changed(update_plot)
    for slider in sliders_c:
        slider.on_changed(update_plot)

    # Instructions
    fig.suptitle(
        (
            'Interactive Step Response Plot\n'
            'Adjust sliders to change parameters p and c'
        ),
        fontsize=12,
        y=0.95
    )

    plt.show()

    return fig, (ax1, ax2), slider_p, sliders_c


if __name__ == "__main__":
    fig, axes, slider_p, sliders_c = create_interactive_step_response_plot()
