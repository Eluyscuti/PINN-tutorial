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
import numpy as np
import time
import matplotlib
from matplotlib import pyplot as pp
import scipy
from scipy.integrate import solve_ivp
import random
import functools
import pickle

import pandas as pd
import functools
from scipy.optimize import curve_fit

# Load the pickle file into a DataFrame



file_path = '/Users/abhi/Desktop/Projects/Sys_ID/Untitled/PINN-tutorial/data/massdamper_data.pkl'

with open(file_path, 'rb') as file:
    data = pickle.load(file)

# Check if it's a dictionary, then print keys
if isinstance(data, dict):
    print(data.keys())
else:
    print("Data is not a dictionary. Type is:", type(data))

#get ground truth data
ground_truth_data = data['data_groundtruth']

print(ground_truth_data.keys())

#get parameters for ground truth data such as x0, v0, t, omega0, and delta
#get the displacement and time data from the no external force model
ground_truth_parameters = ground_truth_data['ground_truth_params']
no_external_force_model = ground_truth_data['model: No external force']

print(ground_truth_parameters.keys())

#map parameters to corresponding variables
ordered_keys = ground_truth_parameters.keys()

(x0, t_span, t_eval, m_true, c_true, k_true, Kp_array, delta_val, omega0_val, T_val) = [ground_truth_parameters[k] for k in ordered_keys]


print(x0)
print(no_external_force_model.keys())

#print(ground_truth_parameters.items())

#get time, velocity, position data from unforced model
t = no_external_force_model['t']
x = no_external_force_model['x']
v = no_external_force_model['v']

times = t
yexact2 = x

#plot our time displacement response
plt.plot(times, yexact2,label="x(t)")
plt.xlabel("t")
plt.ylabel("Amplitude")
plt.title("Underdamped Harmonic Oscillator")
plt.legend()
plt.show()
#print(f"time: {t}")
#print(f"x {x}")
#print(f"v {v}")


#Cool Now lets get into the actual system identification

def get_frf(yexact2, times):
    N = len(yexact2)
    dt = times[1] - times[0]  # Sampling interval
    print(dt)
    yf = np.fft.fft(yexact2)
    xf = np.fft.fftfreq(N, dt)[:N//2] # Get positive frequencies
    print(f"xf: {xf}")

    # 2. Get Magnitude (Power Spectrum)
    magnitude = 2.0/N * np.abs(yf[0:N//2])

    # 2. Get Magnitude (Power Spectrum)
    magnitude = 2.0/N * np.abs(yf[0:N//2])

   
    


    return xf, magnitude

xf, magnitude = get_frf(yexact2, times)

# 3. Find the peak frequency (Experimental wd) 
peak_index = np.argmax(magnitude)
print(peak_index)
wd_experimental = xf[peak_index] * 2 * np.pi # Convert Hz to rad/s
print(f"Experimental Damped Frequency (wd): {wd_experimental} rad/s")

xf_rad = 2* np.pi * xf


plt.figure(figsize=(6,3))
plt.plot(xf * 2 * np.pi, magnitude)
#plt.axvline(wd_experimental, color='r', linestyle='--', label=f'Peak at {wd_experimental:.2f} rad/s')
plt.scatter(wd_experimental, magnitude[peak_index])
plt.title("Frequency Domain (FFT Magnitude)")
plt.xlabel("Frequency (rad/s)")
plt.ylabel("Magnitude")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 4))

# Using bar to show the 'bins'
plt.bar(xf * 2 * np.pi, magnitude, width=0.05, color='tab:blue', alpha=0.7, label='FFT Bins')

# Optional: Add a stem plot on top for clarity
#plt.stem(xf * 2 * np.pi, magnitude, linefmt='r-', markerfmt='ro', basefmt=" ", label='Discrete Frequencies')

plt.title("Discrete Frequency Spectrum (Magnitude)")
plt.xlabel("Frequency (rad/s)")
plt.ylabel("Magnitude")
plt.xlim(0, 5) # Zooming in on the area of interest (near omega0=20)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

print(f"actual wn {omega0_val}")


initial_conditions = x0
def response_magnitude(w, wn, zeta):
    # Fixed initial conditions from your code
    x0, v0 = initial_conditions[0], initial_conditions[1]
   



    numerator = np.sqrt((v0 + 2*zeta*wn*x0)**2 + (w*x0)**2)
    denominator = np.sqrt((wn**2 - w**2)**2 + (2*zeta*wn*w)**2)
    theoretical = numerator / denominator
    
    # Normalize the theoretical curve to its own peak so it's 0 to 1
    return theoretical / np.max(theoretical)

  




mask = (xf_rad > 0) & (xf_rad < 5) # only choosing frequencies between 0 and 5 where the max response magnitude is


w_data = xf_rad[mask]
mag_data = magnitude[mask]
#print(w_data)
#print(mag_data)




popt, _ = curve_fit(response_magnitude, w_data, mag_data, p0=[0.35, 0.1]) #p0 is wn and zeta guess

wn_fit, zeta_fit = popt

zeta_actual = delta_val / omega0_val
wn_actual = omega0_val
print(f"Approximated wn: {wn_fit} rad/s")
print(f"actual wn: {wn_actual} rad/s")
print(f"Approximated zeta: {zeta_fit}")
print(f"actual zeta: {zeta_actual}")

m_est = 1.0
k_est = m_est * (wn_fit**2) #really finding k/m
c_est = 2 * zeta_fit * wn_fit * m_est #really finding c/m

print(f"Calculated Parameters: k={k_est:.2f}, c={c_est:.2f}")