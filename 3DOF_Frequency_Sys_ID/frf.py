import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import solve_ivp
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from torch.optim import lr_scheduler

import time


import scipy
from scipy.integrate import solve_ivp
import random
import functools
import pickle

import pandas as pd
import functools
from scipy.optimize import curve_fit
from scipy import signal, optimize
from scipy.fft import fft, fftfreq

from scipy.linalg import lstsq

from scipy.linalg import inv
from scipy.optimize import fsolve
from scipy.optimize import least_squares
from scipy.stats import trim_mean

import control as ct
from scipy.signal import csd, welch
from scipy.signal import freqresp
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.signal import lsim






def wrap_angle(x):
    """Wrap angle(s) in radians to (-pi, pi]."""
    return (x + np.pi) % (2 * np.pi) - np.pi

file_path = "/Users/abhi/Desktop/Projects/Sys_ID/PINN-tutorial/3DOF_Frequency_Sys_ID/simulation_data_near_trim_nogust_noactuator.csv"

# Load the data
# skipinitialspace=True handles any spaces after the commas in the header
df = pd.read_csv(file_path, skipinitialspace=True)

# Extract into specific variables
t       = df['t'].values
V       = df['V'].values
alpha   = df['alpha'].values
gamma   = df['gamma'].values
Q       = df['Q'].values
delta_T = df['T_act'].values
delta_e = df['elv_act'].values



param_path = "/Users/abhi/Desktop/Projects/Sys_ID/PINN-tutorial/3DOF_Frequency_Sys_ID/truth_model_params_near_trim_nogust_noactuator.csv"

# Load the CSV
param_df = pd.read_csv(param_path)

# Convert to a dictionary for easy access
# We set 'parameter' as the index so the 'value' column maps to the keys
params = param_df.set_index('parameter')['value'].to_dict()

# Assign variables individually
g        = params['g']
rho      = params['rho']
S        = params['S']
cbar     = params['cbar']
m        = params['m']
Jy       = params['Jy']


# Aerodynamic Coefficients
C_L0_actual     = params['C_L0']
C_La_actual     = params['C_La']
C_D0_actual   = params['C_D0']
k_CLCD_actual   = params['k_CLCD']
C_M0_actual     = params['C_M0']
C_Ma_actual     = params['C_Ma']
C_MQ_actual     = params['C_MQ']
C_Me_actual     = params['C_Me']

# Time Constants / Actuator dynamics
tau_T_actual    = params['tau_T']
tau_elv_actual  = params['tau_elv']
tau_gust_actual = params['tau_gust']

print("Aircraft parameters loaded successfully.")
print(f"Mean Chord (cbar): {cbar} m")
print(f"Mass (m): {m} kg")

# ── Load data (replace with your arrays) ──────────────────────────────
# t, V, alpha, gamma, Q, delta_T, delta_e

N  = len(t)
dt = t[1] - t[0]

def deriv(x): return np.gradient(x, dt)
def W(x):     return x * np.hanning(N)   # windowed FFT
def F(x):     return fft(W(x))

freqs = fftfreq(N, dt)

Vdot     = deriv(V)
alphadot = deriv(alpha)
gammadot = deriv(gamma)
Qdot     = deriv(Q)
qhat     = cbar * Q / (2 * V)
q_dyn    = 0.5 * rho * V**2



V_trim     = V[0]
alpha_trim = alpha[0]
gamma_trim = gamma[0]
Q_trim     = Q[0]
delta_T_trim = delta_T[0]
delta_e_trim = delta_e[0]


x0 = np.array([
    V_trim,
    alpha_trim ,
    gamma_trim,
    Q_trim 
])

u0 = np.array([
    delta_T_trim,
    delta_e_trim
])
# 2. Subtract trim to get perturbations (Delta values)
V_delta     = V - V_trim
alpha_delta = alpha - alpha_trim
gamma_delta = gamma - gamma_trim
Q_delta     = Q - Q_trim
dT_delta    = delta_T - delta_T_trim
de_delta    = delta_e - delta_e_trim

# 3. Pass THESE into your FRF function
states_delta = [V_delta, alpha_delta, gamma_delta, Q_delta]
inputs_delta = [dT_delta, de_delta]


state_names = ['V', 'alpha', 'gamma', 'Q']
input_names = ['delta_T', 'delta_e']



print(x0)
print(u0)


def dynamics(x, u, params):
    V, alpha, gamma, Q       = x
    delta_T, delta_e         = u
    CL0, CL_alpha, CD0, k_CD, CM0, CM_alpha, CM_Q, CM_e = params

    V_safe = max(V, 1e-3)
    qhat   = cbar * Q / (2 * V_safe)

    CL = CL0 + CL_alpha * alpha
    CD = CD0 + k_CD * CL**2
    CM = CM0 + CM_alpha * alpha + CM_Q * qhat + CM_e * delta_e

    q_dyn = 0.5 * rho * V**2
    L     = q_dyn * S * CL
    D     = q_dyn * S * CD
    M     = q_dyn * S * cbar * CM
    T     = delta_T

    Vdot     = (-D + T*np.cos(alpha) - m*g*np.sin(gamma)) / m
    gammadot = ( L + T*np.sin(alpha) - m*g*np.cos(gamma)) / (m * V_safe)
    Qdot     = M / Jy
    alphadot = Q - gammadot

    return np.array([Vdot, alphadot, gammadot, Qdot])



def linearise(x0, u0, params, eps=1e-4):
    """
    Computes the A and B matrices for the system dot{x} = Ax + Bu
    around the equilibrium point (x0, u0) using central differences.
    """
    nx = len(x0)
    nu = len(u0)
    
    A = np.zeros((nx, nx))
    B = np.zeros((nx, nu))
    
    # --- Compute A matrix (df/dx) ---
    # We nudge each state variable one by one
    for i in range(nx):
        x_plus = x0.copy()
        x_minus = x0.copy()
        
        x_plus[i] += eps
        x_minus[i] -= eps
        
        # Central difference formula: (f(x+eps) - f(x-eps)) / (2*eps)
        f_plus = dynamics(x_plus, u0, params)
        f_minus = dynamics(x_minus, u0, params)
        
        A[:, i] = (f_plus - f_minus) / (2 * eps)
        
    # --- Compute B matrix (df/du) ---
    # We nudge each input variable one by one
    for j in range(nu):
        u_plus = u0.copy()
        u_minus = u0.copy()
        
        u_plus[j] += eps
        u_minus[j] -= eps
        
        f_plus = dynamics(x0, u_plus, params)
        f_minus = dynamics(x0, u_minus, params)
        
        B[:, j] = (f_plus - f_minus) / (2 * eps)
        
    return A, B

def get_aircraft_tfs(x0, u0, params):
    A, B = linearise(x0, u0, params)
    nx, nu = B.shape
    
    # Pre-define names for clarity
    state_names = ['V', 'alpha', 'gamma', 'Q']
    input_names = ['delta_T', 'delta_e']
    
    tf_matrix = {} # Using a dict for easier access: tfs['delta_e']['Q']
    
    for j in range(nu):
        u_name = input_names[j]
        tf_matrix[u_name] = {}
        
        for i in range(nx):
            x_name = state_names[i]
            
            # SISO selection
            C_single = np.zeros((1, nx))
            C_single[0, i] = 1
            B_single = B[:, [j]]
            
            # Create system - Note: D is usually 0 in aircraft dynamics
            sys_siso = ct.ss(A, B_single, C_single, 0)
            
            # Clean up the TF: removes tiny numerical noise (e.g., 1e-18)
            tf = ct.tf(sys_siso)
            tf_matrix[u_name][x_name] = tf
            
    return tf_matrix, A, B



# Assuming the variables have already been loaded from your dictionary:
# C_L0, C_La, C_D0, k_CLCD, C_M0, C_Ma, C_MQ, C_Me

# Create the theta_true array in the requested order
theta_true = np.array([
    C_L0_actual,     # Lift coefficient at zero alpha
    C_La_actual,     # Lift curve slope
    C_D0_actual,     # Zero-lift drag coefficient
    k_CLCD_actual,   # Induced drag factor (k_CD)
    C_M0_actual,     # Pitching moment at zero alpha
    C_Ma_actual,     # Pitching moment slope (stability)
    C_MQ_actual,     # Pitch damping derivative
    C_Me_actual      # Elevator control authority
])

# Verification print
print("theta_true array:")
print(theta_true)

# If you want to see the values with the parameter names for confirmation:
param_names = ["C_L0", "C_La", "C_D0", "k_CD", "C_M0", "C_Ma", "C_MQ", "C_Me"]
for name, val in zip(param_names, theta_true):
    print(f"{name:8}: {val:>10}")



# Unpack the three return values
tfs_dict, A_mat, B_mat = get_aircraft_tfs(x0, u0, theta_true)

print(A_mat.shape)
print(B_mat.shape)
print(A_mat)
print(B_mat)



for u_name in input_names:
    for x_name in state_names:
        print(f"Transfer Function: {u_name} -> {x_name}")
        # Access using the names (keys) instead of [i][j]
        print(tfs_dict[u_name][x_name]) 
        print("-" * 40)

sys = tfs_dict['delta_e']['Q']

mag, phase, omega = ct.bode_plot(sys, dB=True)
plt.show()

def compute_frfs_basic(t, states_delta, inputs_delta, fs):
    """
    Computes FRFs and Coherence for a 3-DOF aircraft.
    
    States: [V, alpha, gamma, Q] (4 states)
    Inputs: [dT, de] (2 inputs)
    """
    num_states = len(states_delta)
    num_inputs = len(inputs_delta)
    
    # Define nperseg for frequency resolution vs variance tradeoff
    # A common choice is 1/4th or 1/8th of the signal length
    nperseg = min(len(t) // 4, 1024)
    
    # Initialize containers
    # H_data shape: (output_idx, input_idx, freq_idx)
    H_data = []
    coherence = []
    freqs_hz = None

    for i in range(num_states):
        row_H = []
        row_C = []
        for j in range(num_inputs):
            # Extract specific state and input
            y = states_delta[i]
            u = inputs_delta[j]
            
            # Compute Cross-Power Spectral Density P_uy
            f, P_uy = signal.csd(u, y, fs=fs, nperseg=nperseg)
            # Compute Auto-Power Spectral Density P_uu
            _, P_uu = signal.welch(u, fs=fs, nperseg=nperseg)
            # Compute Coherence C_uy
            _, C_uy = signal.coherence(u, y, fs=fs, nperseg=nperseg)
            
            # FRF: H(f) = P_uy / P_uu
            H_ij = P_uy / P_uu
            
            row_H.append(H_ij)
            row_C.append(C_uy)
            freqs_hz = f
            
        H_data.append(row_H)
        coherence.append(row_C)

    return freqs_hz, np.array(H_data), np.array(coherence)



def compute_frfs_MIMO(t, states_delta, inputs_delta, fs):
    """
    Computes MIMO FRFs and Coherence for a 3-DOF aircraft.
    
    Args:
        t: Time vector
        states_delta: List of 4 arrays [V, alpha, gamma, Q]
        inputs_delta: List of 2 arrays [dT, de]
        fs: Sampling frequency
        
    Returns:
        freqs_hz: Frequency axis
        H_data: FRF matrix of shape (num_states, num_inputs, num_freqs)
        coherence: Ordinary coherence matrix of shape (num_states, num_inputs, num_freqs)
    """
    # Convert lists to numpy arrays for easier indexing
    Y = np.asarray(states_delta)  # Shape (4, N)
    U = np.asarray(inputs_delta)   # Shape (2, N)
    
    num_states = Y.shape[0]
    num_inputs = U.shape[0]
    nperseg = min(len(t) // 4, 1024)
    
    # 1. Compute Input Power Spectral Density Matrix S_uu(f)
    # This matrix is (num_inputs, num_inputs, num_freqs)
    # We need to compute cross-spectra between all input pairs
    first_u = U[0]
    f, _ = signal.welch(first_u, fs=fs, nperseg=nperseg)
    num_freqs = len(f)
    
    S_uu = np.zeros((num_inputs, num_inputs, num_freqs), dtype=complex)
    for i in range(num_inputs):
        for j in range(num_inputs):
            f, p_ij = signal.csd(U[i], U[j], fs=fs, nperseg=nperseg)
            S_uu[i, j, :] = p_ij
            
    # 2. Compute Cross-Spectral Density Matrix S_uy(f)
    # This matrix is (num_inputs, num_states, num_freqs)
    S_uy = np.zeros((num_inputs, num_states, num_freqs), dtype=complex)
    for i in range(num_inputs):
        for j in range(num_states):
            f, p_ij = signal.csd(U[i], Y[j], fs=fs, nperseg=nperseg)
            S_uy[i, j, :] = p_ij
            
    # 3. Solve for FRF Matrix H(f)
    # H = S_uu^-1 * S_uy
    # We solve H at every frequency bin
    H_data = np.zeros((num_states, num_inputs, num_freqs), dtype=complex)
    
    for k in range(num_freqs):
        # Extract S_uu and S_uy at this specific frequency
        Suu_k = S_uu[:, :, k]  # (num_inputs, num_inputs)
        Suy_k = S_uy[:, :, k]  # (num_inputs, num_states)
        
        # Use pseudo-inverse or solve to handle potentially singular matrices
        # We want to solve Suu_k * H_k = Suy_k
        try:
            # H_k will be (num_inputs, num_states)
            H_k = np.linalg.solve(Suu_k, Suy_k)
            H_data[:, :, k] = H_k.T
        except np.linalg.LinAlgError:
            # Fallback for singular matrices (e.g., at f=0 or if inputs are perfectly correlated)
            H_k = np.linalg.lstsq(Suu_k, Suy_k, rcond=None)[0]
            H_data[:, :, k] = H_k.T

    # 4. Compute Ordinary Coherence for reference
    coherence = np.zeros((num_states, num_inputs, num_freqs))
    for i in range(num_states):
        for j in range(num_inputs):
            _, C_ij = signal.coherence(U[j], Y[i], fs=fs, nperseg=nperseg)
            coherence[i, j, :] = C_ij

    return f, H_data, coherence




# --- Usage ---
fs = 1.0 / (t[1] - t[0])  # Sampling frequency




freqs_hz, H_data, coherence = compute_frfs_MIMO(t, states_delta, inputs_delta, fs)
omega = 2 * np.pi * freqs_hz  # Convert to rad/s for fitting






def plot_analytical_vs_empirical(freqs_hz, H_data, tfs_dict):
    state_names = ['V', 'alpha', 'gamma', 'Q']
    input_names = ['delta_T', 'delta_e']
    
    # Convert Hz to rad/s for the control library
    w_vec = 2 * np.pi * freqs_hz
    
    fig, axes = plt.subplots(4, 2, figsize=(15, 15), sharex=True)
    
    for j, u_name in enumerate(input_names):
        for i, x_name in enumerate(state_names):
            ax = axes[i, j]
            
            # 1. Plot Empirical Data (FFT)
            mag_fft = 20 * np.log10(np.abs(H_data[i, j, :]) + 1e-9)
            ax.semilogx(freqs_hz, mag_fft, color='gray', alpha=0.3, label='FFT Data')
            
            # 2. Compute and Plot Analytical TF
            sys = tfs_dict[u_name][x_name]
            # mag is returned as absolute value, convert to dB
            mag_ana, _, _ = ct.freqresp(sys, w_vec)
            mag_ana_db = 20 * np.log10(mag_ana)
            
            ax.semilogx(freqs_hz, mag_ana_db, color='tab:red', lw=2, label='Analytical')
            
            # Formatting
            if i == 0: ax.set_title(f"Input: {u_name}")
            if j == 0: ax.set_ylabel(f"{x_name}\nMag (dB)")
            ax.grid(True, which="both", alpha=0.3)
            if i == 0 and j == 0: ax.legend()

    plt.xlabel("Frequency (Hz)")
    plt.tight_layout()
    plt.show()

# --- Run the plot ---
plot_analytical_vs_empirical(freqs_hz, H_data, tfs_dict)




print(f"Mean Power of Elevator Input: {np.mean(np.abs(fft(delta_e - np.mean(delta_e)))**2)}")


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot Elevator Deflection (Input)
ax1.plot(t, delta_e, color='tab:blue', lw=1.5, label='Elevator Deflection ($\delta_e$)')
ax1.set_ylabel('$\delta_e$ [rad]')
ax1.set_title('Elevator Input vs. Time')
ax1.grid(True, which="both", alpha=0.3)
ax1.legend()

# Plot Pitch Rate (Output)
ax2.plot(t, Q, color='tab:orange', lw=1.5, label='Pitch Rate ($Q$)')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('$Q$ [rad/s]')
ax2.set_title('Pitch Rate Response vs. Time')
ax2.grid(True, which="both", alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.show()



def check_mimo_coherence(t, states_delta, inputs_delta, fs):
    """
    Calculates Input-Input Coherence and Multiple Coherence for a MIMO system.
    """
    U = np.asarray(inputs_delta)  # (num_inputs, N)
    Y = np.asarray(states_delta)   # (num_states, N)
    
    num_inputs = U.shape[0]
    num_states = Y.shape[0]
    nperseg = min(len(t) // 4, 1024)

    # 1. Compute Spectral Density Matrices
    # f: frequencies
    # S_uu: (num_inputs, num_inputs, num_freqs)
    # S_yy: (num_states, num_freqs)
    # S_uy: (num_inputs, num_states, num_freqs)
    
    f, _ = signal.welch(U[0], fs=fs, nperseg=nperseg)
    num_freqs = len(f)
    
    S_uu = np.zeros((num_inputs, num_inputs, num_freqs), dtype=complex)
    S_uy = np.zeros((num_inputs, num_states, num_freqs), dtype=complex)
    S_yy = np.zeros((num_states, num_freqs), dtype=complex)

    for i in range(num_inputs):
        for j in range(num_inputs):
            _, S_uu[i, j, :] = signal.csd(U[i], U[j], fs=fs, nperseg=nperseg)
        for j in range(num_states):
            _, S_uy[i, j, :] = signal.csd(U[i], Y[j], fs=fs, nperseg=nperseg)
            
    for i in range(num_states):
        _, S_yy[i, :] = signal.welch(Y[i], fs=fs, nperseg=nperseg)

    # 2. Calculate Input-Input Coherence (dT vs de)
    # Only relevant if num_inputs > 1
    input_coh = np.zeros(num_freqs)
    if num_inputs > 1:
        # gamma^2 = |S_u1u2|^2 / (S_u1u1 * S_u2u2)
        num = np.abs(S_uu[0, 1, :])**2
        den = np.real(S_uu[0, 0, :]) * np.real(S_uu[1, 1, :])
        input_coh = num / den

    # 3. Calculate Multiple Coherence for each state
    # gamma_mult^2 = (H* @ S_uy) / S_yy
    multi_coh = np.zeros((num_states, num_freqs))
    
    for k in range(num_freqs):
        Suu_inv = np.linalg.pinv(S_uu[:, :, k])
        for i in range(num_states):
            Suy_vec = S_uy[:, i, k]
            # Multiple coherence formula for MISO
            # gamma^2 = (Suy.H * Suu_inv * Suy) / Syy
            num = np.conj(Suy_vec).T @ Suu_inv @ Suy_vec
            den = S_yy[i, k]
            multi_coh[i, k] = np.real(num / den)

    return f, input_coh, multi_coh

# Usage:
# f, in_coh, m_coh = check_mimo_coherence(t, states_delta, inputs_delta, fs)

def analyze_and_plot_coherence(t, states_delta, inputs_delta, fs, state_names):
    """
    Computes and plots Input-Input Coherence and Multiple Coherence.
    """
    # Use the diagnostic function from the previous step
    f, in_coh, m_coh = check_mimo_coherence(t, states_delta, inputs_delta, fs)
    
    # 1. Print average values for a quick health check
    print("--- Coherence Health Check (Averages) ---")
    print(f"Input-Input Coherence (dT vs de): {np.mean(in_coh):.3f}")
    if np.mean(in_coh) > 0.8:
        print("WARNING: High input correlation detected. Results may be unreliable.")
        
    for i, name in enumerate(state_names):
        print(f"Multiple Coherence for {name:5}: {np.mean(m_coh[i, :]):.3f}")
    print("------------------------------------------")

    # 2. Plotting for frequency-dependent analysis
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot Input Correlation
    ax[0].semilogx(f, in_coh, label='dT vs de Coherence', color='orange')
    ax[0].set_ylabel('Input Coherence')
    ax[0].set_title('Input Correlation (Should be Low)')
    ax[0].grid(True, which="both", ls="-", alpha=0.5)
    ax[0].axhline(0.8, color='r', linestyle='--', label='Critical Threshold')
    ax[0].legend()

    # Plot Multiple Coherence for each state
    for i in range(m_coh.shape[0]):
        ax[1].semilogx(f, m_coh[i, :], label=f'State: {state_names[i]}')
    
    ax[1].set_ylabel('Multiple Coherence')
    ax[1].set_xlabel('Frequency (Hz)')
    ax[1].set_title('Multiple Coherence (Should be High ~1.0)')
    ax[1].axhline(0.6, color='k', linestyle=':', label='Reliability Limit')
    ax[1].grid(True, which="both", ls="-", alpha=0.5)
    ax[1].legend(loc='lower left')
    
    plt.tight_layout()
    plt.show()

# Example Call:
state_names = ['V', 'alpha', 'gamma', 'Q']
analyze_and_plot_coherence(t, states_delta, inputs_delta, fs, state_names)

import numpy as np
from scipy.optimize import minimize
import control as ct

def fit_cl_alpha(freqs_hz, H_data, coherence, x0, u0, params_dict, input_idx, state_idx, actual_cl_alpha):
    """
    Fits CL_alpha and compares it against the true value.
    """
    # 1. Filter for high-quality data (Coherence > 0.6 and limited frequency range)
    # Most aircraft rigid-body heave dynamics are well-captured below 5-10 Hz
    mask = (coherence[state_idx, input_idx, :] > 0.6) & (freqs_hz < 10.0)
    f_fit = freqs_hz[mask]
    H_exp = H_data[state_idx, input_idx, mask]
    w_fit = 2 * np.pi * f_fit

    if len(f_fit) == 0:
        print("Error: No data points met the coherence threshold. Estimation aborted.")
        return None

    def objective(cl_alpha_guess):
        current_params = params_dict.copy()
        current_params['C_La'] = cl_alpha_guess[0]
        
        param_list = [
            current_params['C_L0'], current_params['C_La'], current_params['C_D0'],
            current_params['k_CLCD'], current_params['C_M0'], current_params['C_Ma'],
            current_params['C_MQ'], current_params['C_Me']
        ]
        
        A, B = linearise(x0, u0, param_list)
        C = np.zeros((1, A.shape[0]))
        C[0, state_idx] = 1
        sys = ct.ss(A, B[:, [input_idx]], C, 0)
        
        # FIX: Unpack 3 values instead of 2
        # mag and phase are returned as arrays matching w_fit
        mag, phase, omega = ct.freqresp(sys, w_fit)
        
        # Reconstruct the complex transfer function H = mag * exp(j * phase)
        # We flatten to ensure they are 1D arrays for the sum
        H_model = mag.flatten() * np.exp(1j * phase.flatten())
        
        channel_error = np.sum(np.abs(np.log10(np.abs(H_exp)) - np.log10(np.abs(H_model)))**2)

        print(f"channel error: {channel_error}")
        # Least squares error on log-magnitude
        return channel_error
    # 2. Optimize
    initial_guess = [params_dict['C_La']] # Start from current nominal value
    res = minimize(objective, initial_guess, method='Nelder-Mead', bounds=[(0.1, 10.0)])
    
    calc_cl_alpha = res.x[0]
    
    # 3. Calculate Percent Error
    # Formula: |(Experimental - Actual) / Actual| * 100
    percent_error = np.abs((calc_cl_alpha - actual_cl_alpha) / actual_cl_alpha) * 100

    print("--- CL_alpha Identification Results ---")
    print(f"True CL_alpha:       {actual_cl_alpha:.4f}")
    print(f"Identified CL_alpha: {calc_cl_alpha:.4f}")
    print(f"Percent Error:       {percent_error:.2f}%")
    print("---------------------------------------")
    
    return calc_cl_alpha


def fit_cl_alpha_multi(freqs_hz, H_data, coherence, x0, u0, params_dict, actual_cl_alpha):
    """
    Fits CL_alpha and compares it against the true value.
    """
    # 1. Filter for high-quality data (Coherence > 0.6 and limited frequency range)
    # Most aircraft rigid-body heave dynamics are well-captured below 5-10 Hz
    good_channels = [(1,3)]
    H_exps_good = []
    w_fit_good = []

    for channel in range(len(good_channels)):
        state_idx = good_channels[channel][1]
        input_idx = good_channels[channel][0]

        mask = (coherence[state_idx, input_idx, :] > 0.6) & (freqs_hz < 10.0)
        f_fit = freqs_hz[mask]
        H_exp = H_data[state_idx, input_idx, mask]
        w_fit = 2 * np.pi * f_fit

        H_exps_good.append(H_exp)
        w_fit_good.append(w_fit)

    if len(f_fit) == 0:
        print("Error: No data points met the coherence threshold. Estimation aborted.")
        return None

    def objective(cl_alpha_guess):
        error = 0
        current_params = params_dict.copy()
        current_params['C_La'] = cl_alpha_guess[0]
        
        param_list = [
            current_params['C_L0'], current_params['C_La'], current_params['C_D0'],
            current_params['k_CLCD'], current_params['C_M0'], current_params['C_Ma'],
            current_params['C_MQ'], current_params['C_Me']
        ]
        
        

        for channel in range(len(good_channels)):
            state_idx = good_channels[channel][1]
            input_idx = good_channels[channel][0]

            A, B = linearise(x0, u0, param_list)
            C = np.zeros((1, A.shape[0]))

            C[0, state_idx] = 1
            sys = ct.ss(A, B[:, [input_idx]], C, 0)
        
            # FIX: Unpack 3 values instead of 2
            # mag and phase are returned as arrays matching w_fit
            mag, phase, omega = ct.freqresp(sys, w_fit[channel])
            
            # Reconstruct the complex transfer function H = mag * exp(j * phase)
            # We flatten to ensure they are 1D arrays for the sum
            H_model = mag.flatten() * np.exp(1j * phase.flatten())
            channel_error = np.sum(np.abs(np.log10(np.abs(H_exp[channel])) - np.log10(np.abs(H_model)))**2)

            print(f"channel {good_channels[channel]} error: {channel_error}")

            error += np.sum(np.abs(np.log10(np.abs(H_exp[channel])) - np.log10(np.abs(H_model)))**2)
        
        # Least squares error on log-magnitude
        print(f"error {error}")
        return error

    # 2. Optimize
    initial_guess = [params_dict['C_La']] # Start from current nominal value
    res = minimize(objective, initial_guess, method='Nelder-Mead', bounds=[(0.1, 10.0)])

  
    
    calc_cl_alpha = res.x[0]
    
    # 3. Calculate Percent Error
    # Formula: |(Experimental - Actual) / Actual| * 100
    percent_error = np.abs((calc_cl_alpha - actual_cl_alpha) / actual_cl_alpha) * 100

    print("--- CL_alpha Identification Results ---")
    print(f"True CL_alpha:       {actual_cl_alpha:.4f}")
    print(f"Identified CL_alpha: {calc_cl_alpha:.4f}")
    print(f"Percent Error:       {percent_error:.2f}%")
    print("---------------------------------------")
    
    return calc_cl_alpha


# --- Usage ---
actual_cl_alpha = theta_true[1] # C_La is the second element
single_ch_cl_alpha = fit_cl_alpha(freqs_hz, H_data, coherence, x0, u0, params, 0, 2, actual_cl_alpha)

multi_ch_cl_alpha = fit_cl_alpha_multi(freqs_hz, H_data, coherence, x0, u0, params, actual_cl_alpha)


