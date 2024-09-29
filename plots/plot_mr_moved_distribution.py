import numpy as np 
import matplotlib.pyplot as plt 
import os 


c = 1.0
epsilon0 = 1.0
mu_0 = 1.0 / (epsilon0 * c**2)
m_unit = 1.0
r_m = 1.0 / 4.0
m_electron = 1 * m_unit
m_ion = m_electron / r_m
t_r = 1.0
r_q = 1.0
n_e = 100 #ここは手動で調整すること
B0 = np.sqrt(n_e) / 1.0
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
sheat_thickness = 2.0 * ion_inertial_length
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
r_e = m_electron * V_Ae / abs(q_electron) / B0

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


fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(111)

mr = "mr4"
dirname = f"/fs51/akutagawakt/PIC/results_mr_{mr}"
savename = "mr4_moved_distance_z_De.png"

record_start = 8000
record_stop = 12000
interval = 80

step = 10000
filename = f"{dirname}/mr_x_electron_{step}.bin"
with open(filename, 'rb') as f:
    x_electron = np.fromfile(f, dtype=np.float32)
x_electron = x_electron.reshape(-1, 3).T
total_electron = x_electron.shape[0]

#target_index = np.where(
#    (59 < x_electron[0] / ion_inertial_length) & (x_electron[0] / ion_inertial_length < 60) &
#    (-0.5 < (x_electron[1] - 0.5 * y_max) / ion_inertial_length) & ((x_electron[1] - 0.5 * y_max) / ion_inertial_length < 0.5)
#)[0]
target_index = np.where(
    (49.5 < x_electron[0] / ion_inertial_length) & (x_electron[0] / ion_inertial_length < 50.5) &
    (-0.5 < (x_electron[1] - 0.5 * y_max) / ion_inertial_length) & ((x_electron[1] - 0.5 * y_max) / ion_inertial_length < 0.5)
)[0]


step = record_start
filename = f"{dirname}/mr_x_electron_{step}.bin"
with open(filename, 'rb') as f:
    x_electron = np.fromfile(f, dtype=np.float32)
x_electron = x_electron.reshape(-1, 3).T

particle_position_start = x_electron[:, target_index]


step = record_stop
filename = f"{dirname}/mr_x_electron_{step}.bin"
with open(filename, 'rb') as f:
    x_electron = np.fromfile(f, dtype=np.float32)
x_electron = x_electron.reshape(-1, 3).T

particle_position_stop = x_electron[:, target_index]


ax1.hist((particle_position_stop[2] - particle_position_start[2]) / r_e, bins=100, color='blue')

ax1.set_xlabel(r'$(z_{150\Omega_{ci}^{-1}} - z_{100\Omega_{ci}^{-1}}) / r_e$', fontsize=20)
ax1.set_ylabel('count', fontsize=20)
ax1.set_xlim(-10.0, 150.0)
ax1.set_ylim(0, 750)

fig.savefig(savename, dpi=200)


