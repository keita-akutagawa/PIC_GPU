import numpy as np 
import matplotlib.pyplot as plt 
import os 


c = 1.0
epsilon0 = 1.0
mu_0 = 1.0 / (epsilon0 * c**2)
m_unit = 1.0
r_m = 1.0 / 2.0
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
sheat_thickness = 10.0 * ion_inertial_length
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


fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(111)

mr = "mr2"
dirname = f"/fs51/akutagawakt/PIC/results_mr_{mr}"
savedir = f"pictures_{mr}"
savename = f"{mr}_orbit_xy.png"

record_start = 2000
record_stop = 8000
interval = 40

step = record_start
filename = f"{dirname}/mr_x_electron_{step}.bin"
with open(filename, 'rb') as f:
    x_electron = np.fromfile(f, dtype=np.float32)
x_electron = x_electron.reshape(-1, 3).T
total_electron = x_electron.shape[0]

target_index = np.where(
    (45 < x_electron[0] / ion_inertial_length) & (x_electron[0] / ion_inertial_length < 55) &
    (-1 < (x_electron[1] - 0.5 * y_max) / ion_inertial_length) & ((x_electron[1] - 0.5 * y_max) / ion_inertial_length < 1)
)[0]

particle_index = target_index[1000:1010]

data_size = np.dtype(np.float32).itemsize
particle_position = np.zeros([len(particle_index), 3, int((record_stop - record_start) / interval + 1)])


for step in range(record_start, record_stop + 1, interval):
    
    filename = f"{dirname}/mr_x_electron_{step}.bin"
    count = 0

    with open(filename, 'rb') as f:
        for index in particle_index:

            x_electron_single = np.zeros(3)
            for i in range(3):
                f.seek(data_size * (index * 3 + i), 0)
                x_electron_single[i] = np.fromfile(f, dtype=np.float32, count=1)[0]

            particle_position[count, :, int((step - record_start) / interval)] = x_electron_single

            count += 1


for i in range(len(particle_index)):
    ax1.scatter(
         particle_position[i, 0, :] / ion_inertial_length, 
        (particle_position[i, 1, :] - 0.5 * (y_max - y_min)) / ion_inertial_length, 
        marker='o', s=5
    )

ax1.set_xlabel('$x / \lambda_i$', fontsize=20)
ax1.set_ylabel('$y / \lambda_i$', fontsize=20)
ax1.set_xlim(0, 100)
ax1.set_ylim(-10, 10)

fig.savefig(savename, dpi=200)

