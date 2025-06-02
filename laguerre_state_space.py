"""
Laguerre Series State Space Representation
Implementation based on Dumont and Servos (1988) paper
"Deterministic adaptive control based on Laguerre series representation"

This module implements the state space model using CasADi symbolic computation
for the Laguerre filter network used in adaptive control.
"""

import casadi as ca
import numpy as np
from typing import Tuple, Optional


class LaguerreStateSpace:
    """
    Implements the Laguerre series state space representation for adaptive control.
    
    The state space model is defined by:
    x(k+1) = A * x(k) + b * u(k)
    y(k) = c^T * x(k)
    
    Where A and b are constructed based on Laguerre filter parameters.
    """
    
    def __init__(self, N: int, p: float, T: float):
        """
        Initialize the Laguerre state space model.
        
        Parameters:
        -----------
        N : int
            Order of the Laguerre filter network (number of filters)
        p : float
            Laguerre parameter (pole location, must be > 0 for stability)
        T : float
            Sampling period
        """
        self.N = N
        self.p = p
        self.T = T
        
        # Calculate Laguerre parameters symbolically
        self.tau1, self.tau2, self.tau3, self.tau4 = self._calculate_tau_parameters()
        
        # Construct state space matrices
        self.A = self._construct_A_matrix()
        self.b = self._construct_b_vector()
        
        # State and input variables
        self.x = ca.SX.sym('x', N)  # State vector
        self.u = ca.SX.sym('u')     # Input scalar
        self.c = ca.SX.sym('c', N)  # Output weight vector
        
        # Define system dynamics
        self.x_next = self.A @ self.x + self.b * self.u
        self.y = self.c.T @ self.x
        
        # Create CasADi functions
        self._create_casadi_functions()
    
    def _calculate_tau_parameters(self) -> Tuple[ca.SX, ca.SX, ca.SX, ca.SX]:
        """
        Calculate the tau parameters from the Laguerre filter theory.
        
        Returns:
        --------
        Tuple of CasADi symbolic expressions for tau1, tau2, tau3, tau4
        """
        # Define symbolic variables for parameters
        p_sym = ca.SX.sym('p')
        T_sym = ca.SX.sym('T')
        
        # Calculate tau parameters as per equations in the paper
        exp_pT = ca.exp(-p_sym * T_sym)
        
        tau1 = exp_pT
        tau2 = T_sym + (2/p_sym) * (exp_pT - 1)
        tau3 = -T_sym * exp_pT - (2/p_sym) * (exp_pT - 1)
        tau4 = ca.sqrt(2 * p_sym * (1 - tau1) / p_sym)
        
        # Substitute actual parameter values
        tau1_val = ca.substitute(tau1, [p_sym, T_sym], [self.p, self.T])
        tau2_val = ca.substitute(tau2, [p_sym, T_sym], [self.p, self.T])
        tau3_val = ca.substitute(tau3, [p_sym, T_sym], [self.p, self.T])
        tau4_val = ca.substitute(tau4, [p_sym, T_sym], [self.p, self.T])
        
        return tau1_val, tau2_val, tau3_val, tau4_val
    
    def _construct_A_matrix(self) -> ca.SX:
        """
        Construct the A matrix based on the Laguerre filter structure.
        
        Returns:
        --------
        ca.SX: N x N state transition matrix
        """
        A = ca.SX.zeros(self.N, self.N)
        
        # Fill the diagonal with tau1
        for i in range(self.N):
            A[i, i] = self.tau1
        
        # Fill the super-diagonal elements
        for i in range(self.N - 1):
            if i == 0:
                # First row, second column
                A[i, i + 1] = 0
            else:
                # Other super-diagonal elements
                A[i, i + 1] = 0
        
        # Fill the sub-diagonal elements
        for i in range(1, self.N):
            A[i, i - 1] = -(self.tau1 * self.tau2 + self.tau3) / self.T
        
        # Fill the first column (except diagonal)
        for i in range(1, self.N):
            A[i, 0] = ((-1)**(i) * self.tau1**(i-1) * (self.tau1 * self.tau2 + self.tau3)) / (self.T**(i))
        
        return A
    
    def _construct_b_vector(self) -> ca.SX:
        """
        Construct the b vector (input matrix).
        
        Returns:
        --------
        ca.SX: N x 1 input vector
        """
        b = ca.SX.zeros(self.N)
        
        # Fill b vector according to equation (9)
        for i in range(self.N):
            if i == 0:
                b[i] = self.tau4
            else:
                b[i] = (-self.tau2 / self.T)**(i) * self.tau4
        
        return b
    
    def _create_casadi_functions(self):
        """Create CasADi functions for efficient computation."""
        
        # State transition function
        self.f_dynamics = ca.Function('f_dynamics', 
                                    [self.x, self.u], 
                                    [self.x_next],
                                    ['x', 'u'], ['x_next'])
        
        # Output function
        self.f_output = ca.Function('f_output',
                                  [self.x, self.c],
                                  [self.y],
                                  ['x', 'c'], ['y'])
        
        # Combined state-space function
        self.f_system = ca.Function('f_system',
                                  [self.x, self.u, self.c],
                                  [self.x_next, self.y],
                                  ['x', 'u', 'c'], ['x_next', 'y'])
    
    def simulate(self, u_seq: np.ndarray, x0: np.ndarray, 
                 c_weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate the Laguerre state space system.
        
        Parameters:
        -----------
        u_seq : np.ndarray
            Input sequence of length K
        x0 : np.ndarray
            Initial state vector of length N
        c_weights : np.ndarray
            Output weight vector of length N
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]: State trajectory (K+1 x N) and output sequence (K x 1)
        """
        K = len(u_seq)
        x_traj = np.zeros((K + 1, self.N))
        y_seq = np.zeros(K)
        
        x_traj[0, :] = x0
        
        for k in range(K):
            # Compute next state
            x_next = self.f_dynamics(x_traj[k, :], u_seq[k])
            x_traj[k + 1, :] = np.array(x_next).flatten()
            
            # Compute output
            y = self.f_output(x_traj[k, :], c_weights)
            y_seq[k] = float(y)
        
        return x_traj, y_seq
    
    def get_system_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get numerical values of the system matrices A and b.
        
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]: A matrix and b vector as numpy arrays
        """
        A_num = np.array(ca.evalf(self.A))
        b_num = np.array(ca.evalf(self.b)).flatten()
        
        return A_num, b_num
    
    def get_tau_values(self) -> Tuple[float, float, float, float]:
        """
        Get numerical values of the tau parameters.
        
        Returns:
        --------
        Tuple[float, float, float, float]: tau1, tau2, tau3, tau4 values
        """
        tau_values = [float(ca.evalf(tau)) for tau in [self.tau1, self.tau2, self.tau3, self.tau4]]
        return tuple(tau_values)


def example_usage():
    """Example demonstrating the usage of the Laguerre state space model."""
    
    # System parameters
    N = 5      # Order of Laguerre network
    p = 2.0    # Laguerre parameter (must be > 0)
    T = 0.1    # Sampling period
    
    # Create Laguerre state space model
    laguerre_sys = LaguerreStateSpace(N, p, T)
    
    # Display tau parameters
    tau1, tau2, tau3, tau4 = laguerre_sys.get_tau_values()
    print(f"Tau parameters:")
    print(f"tau1 = {tau1:.6f}")
    print(f"tau2 = {tau2:.6f}")
    print(f"tau3 = {tau3:.6f}")
    print(f"tau4 = {tau4:.6f}")
    
    # Get system matrices
    A, b = laguerre_sys.get_system_matrices()
    print(f"\nA matrix shape: {A.shape}")
    print(f"A matrix:\n{A}")
    print(f"\nb vector shape: {b.shape}")
    print(f"b vector: {b}")
    
    # Simulation example
    K = 50  # Number of time steps
    u_seq = np.sin(0.1 * np.arange(K))  # Sinusoidal input
    x0 = np.zeros(N)  # Initial state
    c_weights = np.ones(N) / N  # Uniform weights for output
    
    # Run simulation
    x_traj, y_seq = laguerre_sys.simulate(u_seq, x0, c_weights)
    
    print(f"\nSimulation completed:")
    print(f"State trajectory shape: {x_traj.shape}")
    print(f"Output sequence shape: {y_seq.shape}")
    print(f"Final state: {x_traj[-1, :]}")
    print(f"Output range: [{np.min(y_seq):.4f}, {np.max(y_seq):.4f}]")


if __name__ == "__main__":
    example_usage()
