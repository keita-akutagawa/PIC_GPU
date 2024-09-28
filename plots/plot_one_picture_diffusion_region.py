import numpy as np 
import matplotlib.pyplot as plt 
import os 


c = 1.0
epsilon0 = 1.0
mu_0 = 1.0 / (epsilon0 * c**2)
m_unit = 1.0
r_m = 1 / 4
m_electron = 1 * m_unit
m_ion = m_electron / r_m
t_r = 1.0/1.0
r_q = 1.0
n_e = 100 #ここは手動で調整すること
B0 = np.sqrt(n_e) / 1 / np.sqrt(2/(1+t_r))
n_i = int(n_e / r_q)
T_i  = (B0**2 / 2.0 / mu_0) / (n_i + n_e * t_r)
T_e = T_i * t_r
q_unit = np.sqrt(epsilon0 * T_e / n_e) / 5.0
q_electron = -1 * q_unit
q_ion = r_q * q_unit
debye_length = np.sqrt(epsilon0 * T_e / n_e / q_electron**2)
omega_pe = np.sqrt(n_e * q_electron**2 / m_electron / epsilon0)
omega_pi = np.sqrt(n_i * q_ion**2 / m_ion / epsilon0)
omega_ce = q_electron * B0 / m_electron
omega_ci = q_ion * B0 / m_ion
ion_inertial_length = c / omega_pi
sheat_thickness = 1.0 * ion_inertial_length
v_electron = np.array([0.0, 0.0, c * debye_length / sheat_thickness * np.sqrt(2 / (1.0 + 1/t_r))])
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
n_x = int(ion_inertial_length * 100.0)
n_y = int(ion_inertial_length * 50.0)
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


fig = plt.figure(figsize=(18, 6))
ax1 = fig.add_subplot(111)

dirname = "/fs51/akutagawakt/PIC/results_mr_mr4"

step = 10000
savename = f"{step}_mr4_De.png"

filename = f"{dirname}/mr_zeroth_moment_ion_{step}.bin"
with open(filename, 'rb') as f:
    zeroth_moment_ion = np.fromfile(f, dtype=np.float32)
zeroth_moment_ion = zeroth_moment_ion.reshape(n_x, n_y).T
filename = f"{dirname}/mr_zeroth_moment_electron_{step}.bin"
with open(filename, 'rb') as f:
    zeroth_moment_electron = np.fromfile(f, dtype=np.float32)
zeroth_moment_electron = zeroth_moment_electron.reshape(n_x, n_y).T
filename = f"{dirname}/mr_first_moment_ion_{step}.bin"
with open(filename, 'rb') as f:
    first_moment_ion = np.fromfile(f, dtype=np.float32)
first_moment_ion = first_moment_ion.reshape(n_x, n_y, 3).T
filename = f"{dirname}/mr_first_moment_electron_{step}.bin"
with open(filename, 'rb') as f:
    first_moment_electron = np.fromfile(f, dtype=np.float32)
first_moment_electron = first_moment_electron.reshape(n_x, n_y, 3).T

filename = f"{dirname}/mr_B_{step}.bin"
with open(filename, 'rb') as f:
    B = np.fromfile(f, dtype=np.float32)
B = B.reshape(n_x, n_y, 3).T
filename = f"{dirname}/mr_E_{step}.bin"
with open(filename, 'rb') as f:
    E = np.fromfile(f, dtype=np.float32)
E = E.reshape(n_x, n_y, 3).T


current = q_ion * first_moment_ion + q_electron * first_moment_electron
bulk_electron = first_moment_electron / (zeroth_moment_electron + 1e-10)
gamma = np.sqrt(1.0 + (bulk_electron[0]**2 + bulk_electron[1]**2 + bulk_electron[2]**2) / c**2)
De = gamma * (current[0] * (E[0] + bulk_electron[1] * B[2] - bulk_electron[2] * B[1])
            + current[1] * (E[1] + bulk_electron[2] * B[0] - bulk_electron[0] * B[2])
            + current[2] * (E[2] + bulk_electron[0] * B[1] - bulk_electron[1] * B[0]))

X, Y = np.meshgrid(x_coordinate, y_coordinate - y_max / 2) / ion_inertial_length
start_y_enlarged = np.arange(-9.5, 10, 1)
start_points_enlarged1 = np.array([np.ones(start_y_enlarged.shape)*35, start_y_enlarged])
start_points_enlarged2 = np.array([np.ones(start_y_enlarged.shape)*65, start_y_enlarged])

mappable = ax1.pcolormesh(X, Y, De, cmap='jet', vmin=-1.0, vmax=1.0)
cbar = fig.colorbar(mappable, ax=ax1)
cbar.set_label(r'$D_e$', fontsize=24, rotation=90, labelpad=10)
cbar.ax.tick_params(labelsize=20)

ax1.streamplot(X, Y, B[0], B[1], broken_streamlines=False, 
    start_points=start_points_enlarged1.T, color='black', density = 100, linewidth=1, arrowsize=0)
ax1.streamplot(X, Y, B[0], B[1], broken_streamlines=False, 
    start_points=start_points_enlarged2.T, color='black', density = 100, linewidth=1, arrowsize=0)

ax1.set_xlabel('$x / \lambda_i$', fontsize=20)
ax1.set_ylabel('$y / \lambda_i$', fontsize=20)
ax1.set_xlim(30, 70)
ax1.set_ylim(-10, 10)
ax1.tick_params(labelsize=18)

fig.savefig(savename, dpi=200)


