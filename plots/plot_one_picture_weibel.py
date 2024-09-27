import numpy as np 
import matplotlib.pyplot as plt 
import os 


c = 1.0
epsilon0 = 1.0
mu_0 = 1.0 / (epsilon0 * c**2)
m_unit = 1.0
B0 = 1.0
r_m = 1/1
m_electron = 1 * m_unit
m_ion = m_electron / r_m
T_e = 1/2 * m_electron * (0.1 * c)**2
T_i = 1/2 * m_ion * (0.1 * c)**2
C_S = np.sqrt(r_m * T_e)
n_e = 100 #ここは手動で調整すること
q_unit = np.sqrt(T_e / n_e)
r_q = 1.0
q_electron = -1 * q_unit
q_ion = r_q * q_unit
n_i = int(n_e * np.abs(q_electron) / q_ion)
omega_pe = np.sqrt(n_e * q_electron**2 / m_electron / epsilon0)
omega_pi = np.sqrt(n_i * r_m) #直したほうがいい
omega_ce = q_electron * B0 / m_electron
omega_ci = q_ion * B0 / m_ion
V_A = B0 / np.sqrt(mu_0 * (n_e * m_electron + n_i * m_ion))
debye_length = np.sqrt(epsilon0 * T_e / n_e / q_electron**2)
electron_inertial_length = c / omega_pe

dx = debye_length
dy = debye_length
n_x = 256
n_y = 256
x_max = n_x * dx
y_max = n_y * dy
x_coordinate = np.arange(0.0, x_max, dx)
y_coordinate = np.arange(0.0, y_max, dy)
X, Y = np.meshgrid(y_coordinate, x_coordinate)
dt = 0.5
step = 3000
t_max = step * dt
v_ion = 0.0
v_electron = 0.0
v_thermal_ion = np.sqrt(T_i / m_ion)
v_thermal_electron = np.sqrt(T_e / m_electron)

n_ion = int(n_x * n_y * n_i)
n_electron = int(n_x * n_y * n_e)


fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111)

step = 2000

dirname = "/cfca-work/akutagawakt/PIC/double/results_weibel_exact"
savename = f"{step}_weibel_exact.png"
#dirname = "/cfca-work/akutagawakt/PIC/double/results_weibel"
#savename = f"{step}_weibel.png"

filename = f"{dirname}/weibel_B_{step}.bin"
with open(filename, 'rb') as f:
    B = np.fromfile(f, dtype=np.float64)
B = B.reshape(n_x, n_y, 3).T

mappable = ax.pcolormesh(X/electron_inertial_length, Y/electron_inertial_length, np.linalg.norm(B[:, :, :], axis=0), cmap='inferno', vmin=0.0, vmax=3.5)
ax.set_xlabel('$x / \lambda_e$', fontsize=20)
ax.set_ylabel('$y / \lambda_e$', fontsize=20)
ax.set_xlim(0.0, x_max/electron_inertial_length)
ax.set_ylim(0.0, y_max/electron_inertial_length)
ax.tick_params(labelsize=18)
ax.set_aspect('equal')

fig.savefig(savename, dpi=200)


