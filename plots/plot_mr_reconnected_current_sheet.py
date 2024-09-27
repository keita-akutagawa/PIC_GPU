import numpy as np 
import matplotlib.pyplot as plt 
import os 


c = 1.0
epsilon0 = 1.0
mu_0 = 1.0 / (epsilon0 * c**2)
m_unit = 1.0
r_m = 1.0 / 9.0
m_electron = 1 * m_unit
m_ion = m_electron / r_m
t_r = 1.0
r_q = 1.0
n_e = 10 #ここは手動で調整すること
B0 = np.sqrt(n_e) / 1.0
n_i = int(n_e / r_q)
T_i  = (B0**2 / 2.0 / mu_0) / (n_i + n_e * t_r)
T_e = T_i * t_r
q_unit = np.sqrt(epsilon0 * T_e / n_e) / 1.0
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


fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(111)

dirname = "/cfca-work/akutagawakt/PIC/results_mr_thick_small_2"

step = 18000
savename = f"{step}.png"

filename = f"{dirname}/mr_zeroth_moment_ion_{step}.bin"
with open(filename, 'rb') as f:
    zeroth_moment_ion = np.fromfile(f, dtype=np.float32)
zeroth_moment_ion = zeroth_moment_ion.reshape(n_x, n_y).T
filename = f"{dirname}/mr_B_{step}.bin"
with open(filename, 'rb') as f:
    B = np.fromfile(f, dtype=np.float32)
B = B.reshape(n_x, n_y, 3).T

n_i_average = np.sum(zeroth_moment_ion[int(n_y/2), int(n_x/2) - 100 : int(n_x/2) + 100]) / 200.0
omega_pi_average = np.sqrt(n_i_average * q_ion**2 / m_ion / epsilon0)
ion_inertial_length_average = c / omega_pi_average
Bx_average = np.sum(B[0, :, int(n_x/2) - 100 : int(n_x/2) + 100], axis=1) / 200.0
print(ion_inertial_length, ion_inertial_length_average)

ax1.plot((y_coordinate - 0.5 * (y_max - y_min)) / ion_inertial_length, Bx_average, label="simulation")
for i in range(1, 8):
    ax1.plot((y_coordinate - 0.5 * (y_max - y_min)) / ion_inertial_length, 
              B0 * np.tanh((y_coordinate - 0.5 * (y_max - y_min)) / (i * ion_inertial_length_average)), 
              label=f"thickness = {i}" + r"$\lambda_i$")

ax1.set_xlim(-20, 20)
ax1.legend(loc='upper right')

fig.savefig(savename, dpi=200)



