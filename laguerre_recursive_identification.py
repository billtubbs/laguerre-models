import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from itertools import pairwise


class LaguerreRecursiveIdentifier:
    """
    Recursive identification of Laguerre-based dynamic model using CasADi
    Based on Zervos & Dumont (1988) approach
    """
    
    def __init__(self, n_order=3, initial_p=0.5, initial_c=None, 
                 forgetting_factor=0.98, initial_P_scale=100.0):
        """
        Initialize the recursive identifier
        
        Args:
            n_order: Order of Laguerre model (number of basis functions)
            initial_p: Initial estimate of Laguerre pole parameter
            initial_c: Initial estimate of output coefficients
            forgetting_factor: Forgetting factor for RLS (0 < lambda < 1)
            initial_P_scale: Initial covariance matrix scaling
        """
        self.n_order = n_order
        self.lambda_f = forgetting_factor
        
        # Initialize parameters
        self.p_est = initial_p
        if initial_c is None:
            self.c_est = np.ones(n_order) / n_order  # Equal weights initially
        else:
            self.c_est = np.array(initial_c)
            
        # Initialize Laguerre states
        self.l_states = np.zeros(n_order)
        
        # RLS parameters for coefficient estimation
        self.P_c = np.eye(n_order) * initial_P_scale
        
        # Storage for adaptation history
        self.history = {
            'p_estimates': [self.p_est],
            'c_estimates': [self.c_est.copy()],
            'prediction_errors': [],
            'outputs': [],
            'predictions': []
        }
        
        # CasADi setup for parameter optimization
        self._setup_casadi_optimizer()
        
    def _setup_casadi_optimizer(self):
        """Setup CasADi optimizer for pole parameter estimation"""
        # Decision variable (pole parameter p)
        self.p_var = ca.MX.sym('p')
        
        # Parameters for the optimization problem
        self.u_data = ca.MX.sym('u_data', self.n_order)  # Recent input history
        self.y_target = ca.MX.sym('y_target')  # Target output
        self.dt = ca.MX.sym('dt')  # Time step
        
        # Laguerre state dynamics (simplified for optimization)
        l_pred = self._predict_laguerre_states(self.p_var, self.u_data, self.dt)
        y_pred = ca.dot(self.c_est, l_pred)
        
        # Objective: minimize prediction error
        objective = (y_pred - self.y_target)**2
        
        # Constraints: 0 < p < 1 for stability
        constraints = []
        lbg = []
        ubg = []
        
        # Setup NLP
        nlp = {
            'x': self.p_var,
            'p': ca.vertcat(self.u_data, self.y_target, self.dt),
            'f': objective,
            'g': ca.vertcat(*constraints) if constraints else ca.MX()
        }
        
        # Solver options
        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.max_iter': 50
        }
        
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        
    def _predict_laguerre_states(self, p, u_history, dt):
        """Predict Laguerre states using CasADi operations"""
        sqrt_2p = ca.sqrt(2 * p)
        l = ca.MX.zeros(self.n_order)
        
        # Simplified state prediction (assuming zero initial conditions)
        # This is an approximation for the optimization
        for i in range(self.n_order):
            if i == 0:
                l[i] = sqrt_2p * u_history[0] * dt / (1 + p * dt)
            else:
                l[i] = sqrt_2p * p * l[i-1] * dt / (1 + p * dt)
                
        return l
    
    def update_laguerre_states(self, u, dt):
        """Update Laguerre states using Euler integration"""
        sqrt_2p = np.sqrt(2 * self.p_est)
        
        # State derivatives
        dl_dt = np.zeros(self.n_order)
        dl_dt[0] = -self.p_est * self.l_states[0] + sqrt_2p * u
        
        for i in range(1, self.n_order):
            dl_dt[i] = (sqrt_2p * self.p_est * self.l_states[i-1] - 
                       self.p_est * self.l_states[i])
        
        # Euler integration
        self.l_states += dl_dt * dt
        
    def predict_output(self):
        """Predict output based on current states and coefficients"""
        return np.dot(self.c_est, self.l_states)
    
    def update_coefficients_rls(self, y_measured):
        """Update output coefficients using Recursive Least Squares"""
        # Regression vector (current Laguerre states)
        phi = self.l_states.reshape(-1, 1)
        
        # Prediction error
        y_pred = self.predict_output()
        error = y_measured - y_pred
        
        # RLS update
        P_phi = self.P_c @ phi
        denominator = self.lambda_f + phi.T @ P_phi
        K = P_phi / denominator  # Kalman gain
        
        # Update coefficients
        self.c_est += (K * error).flatten()
        
        # Update covariance matrix
        self.P_c = (self.P_c - K @ phi.T @ self.P_c) / self.lambda_f
        
        return error
    
    def update_pole_parameter(self, u_history, y_target, dt):
        """Update pole parameter using CasADi optimization"""
        try:
            # Prepare parameters for optimization
            p_params = np.concatenate([u_history, [y_target], [dt]])
            
            # Solve optimization problem
            solution = self.solver(
                x0=self.p_est,
                p=p_params,
                lbx=0.01,  # Lower bound for p
                ubx=0.99   # Upper bound for p
            )
            
            # Update pole estimate
            if self.solver.stats()['success']:
                self.p_est = float(solution['x'])
                # Apply bounds for safety
                self.p_est = np.clip(self.p_est, 0.01, 0.99)
                
        except Exception as e:
            # If optimization fails, keep previous estimate
            print(f"Optimization failed: {e}")
    
    def recursive_update(self, u, y_measured, dt=0.1, update_pole_every=10):
        """
        Perform one step of recursive identification
        
        Args:
            u: Current input value
            y_measured: Current measured output
            dt: Time step
            update_pole_every: Update pole parameter every N steps
        """
        # Update Laguerre states
        self.update_laguerre_states(u, dt)
        
        # Predict output
        y_pred = self.predict_output()
        
        # Update coefficients using RLS
        error = self.update_coefficients_rls(y_measured)
        
        # Periodically update pole parameter
        if len(self.history['prediction_errors']) % update_pole_every == 0:
            # Use recent input history for pole optimization
            u_history = np.ones(self.n_order) * u  # Simplified
            self.update_pole_parameter(u_history, y_measured, dt)
        
        # Store history
        self.history['p_estimates'].append(self.p_est)
        self.history['c_estimates'].append(self.c_est.copy())
        self.history['prediction_errors'].append(float(error))
        self.history['outputs'].append(y_measured)
        self.history['predictions'].append(y_pred)
        
        return y_pred, error


def make_input_step_functon(t_start, t_stop, u0=0.0, n_steps=9):
    """Generate function that returns values from a prbs-like input signal"""
    t_steps = np.linspace(t_start, t_stop, n_steps, endpoint=False)
    step_data = np.array([1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1], dtype='float')
    assert step_data.shape[0] >= t_steps.shape[0]

    def step_function(t):
        if t < t_start:
            return u0
        i = np.searchsorted(t_steps, t)
        if i == n_steps:
            return step_data[-1]
        return step_data[i]
    
    return step_function


def generate_test_data(t_span, true_p=0.6, true_c=(0.8, 0.5, 0.2), noise_level=0.05):
    """Generate test data from a true Laguerre system"""
    true_c = np.array(true_c)
    n_order = true_c.shape[0]

    step_function = make_input_step_functon(t_span[-1] / 15, t_span[-1], n_steps=14)

    def laguerre_system(states, t):
        """True system dynamics"""
        #u = 1.0 if t >= 1.0 else 0.0  # Step input at t=1
        u = step_function(t)

        sqrt_2p = np.sqrt(2 * true_p)
        
        dl_dt = np.zeros(n_order)
        dl_dt[0] = -true_p * states[0] + sqrt_2p * u
        
        for i in range(1, n_order):
            dl_dt[i] = sqrt_2p * true_p * states[i-1] - true_p * states[i]
            
        return dl_dt
    
    # Solve ODE
    initial_states = np.zeros(n_order)
    states_history = odeint(laguerre_system, initial_states, t_span)
    
    # Generate outputs
    y_true = states_history @ true_c

    # Generate inputs
    u_data = np.array([step_function(t) for t in t_span])

    # Add noise
    y_measured = y_true + np.random.normal(0, noise_level, len(y_true))
    
    return u_data, y_measured, y_true


def main():
    """Main function to demonstrate recursive identification"""
    # Time vector
    t_span = np.linspace(0, 100, 1001)
    dt = t_span[1] - t_span[0]
    
    # True system parameters
    true_p = 0.5
    true_c = np.array([0.8, -0.5, -0.1])
    
    # Generate test data
    print("Generating test data...")
    u_data, y_measured, y_true = generate_test_data(t_span, true_p, true_c)
    
    # Initialize identifier with different initial estimates
    identifier = LaguerreRecursiveIdentifier(
        n_order=3,
        initial_p=0.3,  # Wrong initial guess
        initial_c=[0.5, 0.5, 0.5],  # Wrong initial guess
        forgetting_factor=0.98
    )
    
    print("Running recursive identification...")
    
    # Run recursive identification
    y_predictions = []
    for i, (u, y) in enumerate(zip(u_data, y_measured)):
        y_pred, error = identifier.recursive_update(u, y, dt)
        y_predictions.append(y_pred)
        
        if i % 100 == 0:
            print(f"Step {i}: p_est = {identifier.p_est:.3f}, "
                  f"c_est = {identifier.c_est}, error = {error:.4f}")
    
    # Results
    print("\nFinal Results:")
    print(f"True p: {true_p:.3f}, Estimated p: {identifier.p_est:.3f}")
    print(f"True c: {true_c}")
    print(f"Estimated c: {identifier.c_est}")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Output comparison
    axes[0, 0].plot(t_span, y_true, 'b-', label='True output', linewidth=2)
    axes[0, 0].plot(t_span, y_measured, 'r.', label='Measured (noisy)', alpha=0.5)
    axes[0, 0].plot(t_span, y_predictions, 'g--', label='Predicted', linewidth=2)
    axes[0, 0].set_xlabel('Time [s]')
    axes[0, 0].set_ylabel('Output')
    axes[0, 0].set_title('Output Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Parameter convergence
    p_history = identifier.history['p_estimates']
    axes[0, 1].plot(p_history, 'b-', linewidth=2)
    axes[0, 1].axhline(y=true_p, color='r', linestyle='--', label=f'True p = {true_p}')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Pole Parameter p')
    axes[0, 1].set_title('Pole Parameter Convergence')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Coefficient convergence
    c_history = np.array(identifier.history['c_estimates'])
    for i in range(3):
        axes[1, 0].plot(c_history[:, i], label=f'c[{i}]', linewidth=2)
        axes[1, 0].axhline(y=true_c[i], color=f'C{i}', linestyle='--', alpha=0.7)
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Coefficients')
    axes[1, 0].set_title('Coefficient Convergence')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Prediction error
    error_history = identifier.history['prediction_errors']
    axes[1, 1].plot(error_history, 'r-', linewidth=1)
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Prediction Error')
    axes[1, 1].set_title('Prediction Error Evolution')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Calculate final errors
    p_error = abs(identifier.p_est - true_p) / true_p * 100
    c_error = np.mean(abs(identifier.c_est - true_c) / abs(true_c) * 100)
    
    print(f"\nFinal estimation errors:")
    print(f"Pole parameter error: {p_error:.2f}%")
    print(f"Average coefficient error: {c_error:.2f}%")

if __name__ == "__main__":
    main()