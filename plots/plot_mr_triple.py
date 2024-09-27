import numpy as np 
import matplotlib.pyplot as plt 
import os 


c = 1.0
epsilon0 = 1.0
mu_0 = 1.0 / (epsilon0 * c**2)
m_unit = 1.0
r_m = 1 / 1
m_electron = 1 * m_unit
m_ion = m_electron / r_m
t_r = 1.0 / 1.0
r_q = 1.0
n_i = 100
n_heavy = 10
n_e = 100
B0 = np.sqrt(n_e) / 1.0 / np.sqrt(2 / (1 + t_r))
T_i  = (B0**2 / 2.0 / mu_0) / (n_i + n_e * t_r)
T_e = T_i * t_r
q_unit = np.sqrt(epsilon0 * T_e / n_e)
q_electron = -1 * q_unit
q_ion = r_q * q_unit
debye_length = np.sqrt(epsilon0 * T_e / n_e / q_electron**2)
omega_pe = np.sqrt(n_e * q_electron**2 / m_electron / epsilon0)
omega_pi = np.sqrt(n_i * q_ion**2 / m_ion / epsilon0)
omega_ce = q_electron * B0 / m_electron
omega_ci = q_ion * B0 / m_ion
ion_inertial_length = c / omega_pi
sheat_thickness = 5.0 * ion_inertial_length
v_electron = np.array([0.0, 0.0, c * debye_length / sheat_thickness * np.sqrt(2 / (1.0 + 1 / t_r))])
v_ion = -v_electron / t_r
v_thermal_electron = np.sqrt(2.0 * T_e / m_electron)
v_thermal_ion = np.sqrt(2.0 * T_i / m_ion)
v_thermal_electron_background = np.sqrt(2.0 * T_e * 0.2 / m_electron)
v_thermal_ion_background = np.sqrt(2.0 * T_i * 0.2 / m_ion)
V_Ai = B0 / np.sqrt(mu_0 * n_i * m_ion)
V_Ae = B0 / np.sqrt(mu_0 * n_e * m_electron)
beta_e = n_e * T_e / (B0**2 / 2 / mu_0)
beta_i = n_i * T_i / (B0**2 / 2 / mu_0)

dx = 1.0
dy = 1.0
n_x = int(ion_inertial_length * 1000.0)
n_y = int(ion_inertial_length * 500.0)
x_min = 0.0 * dx
y_min = 0.0 * dy
x_max = n_x * dx
y_max = n_y * dy
x_coordinate = np.arange(0.0, x_max, dx)
y_coordinate = np.arange(0.0, y_max, dy)
dt = 0.5
step = 10000
t_max = step * dt

n_ion = int(n_x * n_i * 2.0 * sheat_thickness)
n_electron = int(n_ion * abs(q_ion / q_electron))
n_ion_background = int(n_x * n_y * 0.2 * n_i)
n_electron_background = int(n_x * n_y * 0.2 * n_e)
n_particle = n_ion + n_ion_background + n_electron + n_electron_background

fig = plt.figure(figsize=(18, 12))
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)

dirname = "/cfca-work/akutagawakt/PIC_triple/results_5:15:20"
savedir = "pictures_triple_5:15:20"

kernel_size = 3

def apply_convolution(data, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
    return np.convolve(data.flatten(), kernel.flatten(), mode='same').reshape(data.shape)


start_y_enlarged = np.arange(200, 301, 5)
start_points_enlarged = np.array([np.ones(start_y_enlarged.shape)*500.0, start_y_enlarged])

for step in range(0, 12000+1, 80):
    ax1.clear()
    ax2.clear()

    savename = f"{savedir}/{step}.png"

    filename = f"{dirname}/mr2008_zeroth_moment_ion_{step}.bin"
    with open(filename, 'rb') as f:
        zeroth_moment_ion = np.fromfile(f, dtype=np.float32)
    zeroth_moment_ion = zeroth_moment_ion.reshape(n_x, n_y).T
    filename = f"{dirname}/mr2008_zeroth_moment_electron_{step}.bin"
    with open(filename, 'rb') as f:
        zeroth_moment_electron = np.fromfile(f, dtype=np.float32)
    zeroth_moment_electron = zeroth_moment_electron.reshape(n_x, n_y).T
    filename = f"{dirname}/mr2008_zeroth_moment_heavy_ion_{step}.bin"
    with open(filename, 'rb') as f:
        zeroth_moment_heavy_ion = np.fromfile(f, dtype=np.float32)
    zeroth_moment_heavy_ion = zeroth_moment_heavy_ion.reshape(n_x, n_y).T
    filename = f"{dirname}/mr2008_B_{step}.bin"
    with open(filename, 'rb') as f:
        B = np.fromfile(f, dtype=np.float32)
    B = B.reshape(n_x, n_y, 3).T

    X, Y = np.meshgrid(x_coordinate, y_coordinate) / ion_inertial_length

    zeroth_moment_ion_smoothed = apply_convolution(zeroth_moment_ion, kernel_size)
    zeroth_moment_electron_smoothed = apply_convolution(zeroth_moment_electron, kernel_size)
    zeroth_moment_heavy_ion_smoothed = apply_convolution(zeroth_moment_heavy_ion, kernel_size)

    X, Y = np.meshgrid(x_coordinate, y_coordinate) / ion_inertial_length

    mappable = ax1.pcolormesh(X, Y, zeroth_moment_ion_smoothed / n_i, cmap='jet', vmin=0.2, vmax=1.2)
    if step == 0:
        cbar = fig.colorbar(mappable, ax=ax1, pad=0.02, shrink=0.9, aspect=10, ticks=np.arange(0.2, 1.3, 0.2))
        cbar.set_label(r'$n_i / n_{i0}$', fontsize=24)
        cbar.ax.tick_params(labelsize=20)
    #ax1.streamplot(X, Y, B[0], B[1], broken_streamlines=False, 
    #    start_points=start_points_enlarged.T, color='white', density = 100, linewidth=1, arrowsize=0)
    #ax1.set_xlabel('$x / \lambda_i$', fontsize=20)
    ax1.set_ylabel('$y / \lambda_i$', fontsize=20)
    ax1.set_xlim(0, 1000)
    ax1.set_ylim(200, 300)
    ax1.tick_params(labelsize=18)
    ax1.text(0.5, 1.05, f"{step / (1.0 / omega_ci / dt):.2f}" + r" $\Omega_{{ci}}^{{-1}}$", ha='center', transform=ax1.transAxes, fontsize=32)

    mappable = ax2.pcolormesh(X, Y, zeroth_moment_electron_smoothed / n_e, cmap='jet', vmin=0.2, vmax=1.2)
    if step == 0:
        cbar = fig.colorbar(mappable, ax=ax2, pad=0.02, shrink=0.9, aspect=10, ticks=np.arange(0.2, 1.3, 0.2))
        cbar.set_label(r'$n_e / n_{e0}$', fontsize=24)
        cbar.ax.tick_params(labelsize=20)
    #ax2.streamplot(X, Y, B[0], B[1], broken_streamlines=False, 
    #    start_points=start_points_enlarged.T, color='white', density = 100, linewidth=1, arrowsize=0)
    #ax2.set_xlabel('$x / \lambda_i$', fontsize=20)
    ax2.set_ylabel('$y / \lambda_i$', fontsize=20)
    ax2.set_xlim(0, 1000)
    ax2.set_ylim(200, 300)
    ax2.tick_params(labelsize=18)

    mappable = ax3.pcolormesh(X, Y, zeroth_moment_heavy_ion_smoothed / n_heavy, cmap='jet', vmin=0.0, vmax=2.0)
    if step == 0:
        cbar = fig.colorbar(mappable, ax=ax3, pad=0.02, shrink=0.9, aspect=10, ticks=np.arange(0.0, 2.6, 0.5))
        cbar.set_label(r'$n_{hi} / n_{hi0}$', fontsize=24)
        cbar.ax.tick_params(labelsize=20)
    #ax3.streamplot(X, Y, B[0], B[1], broken_streamlines=False, 
    #    start_points=start_points_enlarged.T, color='white', density = 100, linewidth=1, arrowsize=0)
    ax3.set_xlabel('$x / \lambda_i$', fontsize=20)
    ax3.set_ylabel('$y / \lambda_i$', fontsize=20)
    ax3.set_xlim(0, 1000)
    ax3.set_ylim(200, 300)
    ax3.tick_params(labelsize=18)

    fig.savefig(savename, dpi=200)


