"""
@author: Brian Guiana

Electromagnetic equations used are from [2] with some notation from [3].
Speckling code performed by [7]. The class [4] was used frequently in all
aspects.

[2] D.M. Pozar, Microwave Electronics. Wiley, USA, 4th edition, 2012.

[3] C.A. Balanis, Advanced Engineering Electromagnetics.
  Wiley, USA, 1st edition, 1989.

[4] ECE 504: Machine Learning for Electromagnetics.
  University Course, Spring 2022, University of Idaho, Moscow, ID.

[7] Pyspeckle, https://pyspeckle2.readthedocs.io/en/latest/#}.
  Accessed: Oct. 15, 2021.

"""

import numpy as np
import pyspeckle as psk
from time import time


# =============================================================================
# Input Parameters
# =============================================================================

SNR = 0.01                   # Signal to noise ratio (V/m // V/m)
NF = 4                       # Noise figure, choose from below
                             #     0: No noise (Control)
                             #     1: Uniform noise
                             #     2: exponential noise
                             #     3: gaussian noise
                             #     4: correlated noise

num_samples = 10000            # Total samples in the data set (samples)
pixels = 30                  # Image size (pixels x pixels)
a = 1.07e-2                  # Waveguide width (m)
b = 0.43e-2                  # Waveguide height (m)
f = 60e9                     # Source frequency (Hz)
eps_rel = 4.0                # Relative permittivity
A = 1                        # TE mode base amplitude
B = 1                        # TM mode base amplitude

# =============================================================================
# Automated data generation
# =============================================================================

rng = np.random.default_rng()
nx = pixels
ny = pixels
x = np.linspace(0, a, nx)
y = np.linspace(0, b, ny)
x, y = np.meshgrid(x, y)
omega = 2 * np.pi * f
mu = 4e-7 * np.pi
eps = eps_rel * 1e-9/(36*np.pi)
k = omega * np.sqrt(mu * eps)
mag_Ex = np.empty([num_samples, nx*ny])
ph_Ex = np.empty([num_samples, nx*ny])
mag_Ey = np.empty([num_samples, nx*ny])
ph_Ey = np.empty([num_samples, nx*ny])
mode = np.empty(num_samples, dtype=str)
mode_m = np.empty(num_samples)
mode_n = np.empty(num_samples)
i = 0
s = 0
start_time = time()
print('starting\n')
while i < num_samples:
    try:
        m = rng.integers(4)
        n = rng.integers(4)
        mode_m[i] = m
        mode_n[i] = n
        SWX = m * np.pi * x / a
        SWY = n * np.pi * y / b
        kc = np.sqrt((m * np.pi / a)**2 + (n * np.pi / b)**2)
        beta = np.sqrt(k**2 - kc**2)
        valid = m or n
        if m < 1 or n < 1:
            raise
        select_mode = rng.integers(2)
        if select_mode == 0:
            mode[i] = 'E'
            AE = 1j * omega * mu * np.pi / kc**2 * A
            AH = 1j * beta * np.pi / kc**2 * A
            Ex = AE * (n / b) * np.cos(SWX) * np.sin(SWY)
            Ey = AE * (-1 * m / a) * np.sin(SWX) * np.cos(SWY)
        elif select_mode == 1:
            mode[i] = 'M'
            BE = -1j * beta * np.pi / kc**2 * B
            BH = 1j * omega * eps * np.pi / kc**2 * B
            Ex = BE * (m / a) * np.cos(SWX) * np.sin(SWY)
            Ey = BE * (n / b) * np.sin(SWX) * np.cos(SWY)

        if NF == 0:
            rx = np.zeros_like(Ex)
            ix = np.zeros_like(Ex)
            ry = np.zeros_like(Ex)
            iy = np.zeros_like(Ex)
        elif NF == 1:
            rx = rng.uniform(0, 1, (nx, ny)) - 0.5
            ix = rng.uniform(0, 1, (nx, ny)) - 0.5
            ry = rng.uniform(0, 1, (nx, ny)) - 0.5
            iy = rng.uniform(0, 1, (nx, ny)) - 0.5
        elif NF == 2:
            rx = rng.exponential(1, (nx, ny))
            ix = rng.exponential(1, (nx, ny))
            ry = rng.exponential(1, (nx, ny))
            iy = rng.exponential(1, (nx, ny))
        elif NF == 3:
            rx = rng.normal(0, 1, (nx, ny))
            ix = rng.normal(0, 1, (nx, ny))
            ry = rng.normal(0, 1, (nx, ny))
            iy = rng.normal(0, 1, (nx, ny))
        elif NF == 4:
            rx = psk.create_Exponential(nx, pix_per_speckle=4, shape='square', alpha=a/b) - 0.5
            ix = psk.create_Exponential(nx, pix_per_speckle=4, shape='square', alpha=a/b) - 0.5
            ry = psk.create_Exponential(nx, pix_per_speckle=4, shape='square', alpha=a/b) - 0.5
            iy = psk.create_Exponential(nx, pix_per_speckle=4, shape='square', alpha=a/b) - 0.5
        nfx = (rx + 1j*ix)/SNR
        nfy = (ry + 1j*iy)/SNR
        Ex = Ex + np.mean(np.abs(Ex))*nfx
        Ey = Ey + np.mean(np.abs(Ey))*nfy
        mag_Ex[i] = abs(Ex.reshape(nx*ny))/np.max(np.abs(Ex))
        ph_Ex[i] = np.angle(Ex.reshape(nx*ny))/np.pi
        mag_Ey[i] = abs(Ey.reshape(nx*ny))/np.max(np.abs(Ey))
        ph_Ey[i] = np.angle(Ey.reshape(nx*ny))/np.pi
        if not i % 50:
            current_time = time()
            time_elapsed = current_time - start_time
            time_per_step = time_elapsed / (i+1)
            time_remaining = (num_samples - i-1) * time_per_step
            print('Data generated: {}\t Time remaining: {:.4f} s'.format(i, time_remaining))
        i += 1
    except:
        s += 1
        pass

# =============================================================================
# Postprocessing / saving
# =============================================================================

print('\n\n{} invalid samples discarded during generation'.format(s))
print('Moving valid data to a dictionary...')
dataset = {'mag_Ex': mag_Ex,
           'ph_Ex': ph_Ex,
           'mag_Ey': mag_Ey,
           'ph_Ey': ph_Ey,
           'mode': mode,
           'm': mode_m,
           'n': mode_n}

print('\nSaving dictionary')
if NF == 0:
    name = 'control_{}'.format(SNR)
    np.save('./../results/control_{}'.format(SNR), dataset, allow_pickle=True)
elif NF == 1:
    name = 'uniform_{}'.format(SNR)
    np.save('./../results/uniform_{}'.format(SNR), dataset, allow_pickle=True)
elif NF == 2:
    name = 'exponential_{}'.format(SNR)
    np.save('./../results/exponential_{}'.format(SNR), dataset, allow_pickle=True)
elif NF == 3:
    name = 'normal_{}'.format(SNR)
    np.save('./../results/normal_{}'.format(SNR), dataset, allow_pickle=True)
elif NF == 4:
    name = 'correlated_{}'.format(SNR)
    np.save('./../results/correlated_{}'.format(SNR), dataset, allow_pickle=True)
print('Saved to Results as {}.npy'.format(name))
