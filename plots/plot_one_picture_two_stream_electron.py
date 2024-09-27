import numpy as np 
import matplotlib.pyplot as plt 
import os 


c = 1.0
epsilon0 = 1.0
mu_0 = 1.0 / (epsilon0 * c**2)
m_unit = 1.0
r_m = 1/100
t_r = 1/100
m_electron = 1 * m_unit
m_ion = m_electron / r_m
r_q = 1.0
T_e = 1/2 * m_electron * (0.01*c)**2
T_i = T_e / t_r
n_e = 100
B0 = np.sqrt(n_e) / 10.0
q_unit = np.sqrt(epsilon0 * T_e / n_e)
q_electron = -1 * q_unit
q_ion = r_q * q_unit
n_i = int(n_e * np.abs(q_electron) / q_ion)
omega_pe = np.sqrt(n_e * q_electron**2 / m_electron / epsilon0)
omega_pi = np.sqrt(n_i * q_ion**2 / m_ion / epsilon0)
omega_ce = q_electron * B0 / m_electron
omega_ci = q_ion * B0 / m_ion
V_A = c * np.sqrt(B0**2 / (n_e*m_electron + n_i*m_ion))
C_S = np.sqrt(r_m * T_e)
debye_length = np.sqrt(epsilon0 * T_e / n_e / q_electron**2)

dx = debye_length
n_x = 512
x_max = n_x * dx
x_coordinate = np.arange(0.0, x_max, dx)
dt = 0.5
step = 10000
t_max = step * dt
v_thermal_ion = np.sqrt(2.0 * T_i / m_ion)
v_thermal_electron = np.sqrt(2.0 * T_e / m_electron)
v_ion = np.array([0.0, 0.0, 0.0])
v_electron = np.array([-10.0*v_thermal_ion, 0.0, 0.0])
v_beam = np.array([10.0*v_thermal_ion, 0.0, 0.0])

n_ion = int(n_x * n_i)
n_electron = int(n_x * n_e / 2)
n_beam = int(n_x * n_e / 2)


fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111)

dirname = "/cfca-work/akutagawakt/PIC/results_two_stream_electron"

step = 3000
savename = f"{step}.png"

filename = f"{dirname}/two_stream_electron_x_ion_{step}.bin"
with open(filename, 'rb') as f:
    x_ion = np.fromfile(f, dtype=np.float32)
x_ion = x_ion.reshape(n_ion, 3).T
filename = f"{dirname}/two_stream_electron_x_electron_{step}.bin"
with open(filename, 'rb') as f:
    x_electron = np.fromfile(f, dtype=np.float32)
x_electron = x_electron.reshape(n_electron+n_beam, 3).T
filename = f"{dirname}/two_stream_electron_v_ion_{step}.bin"
with open(filename, 'rb') as f:
    v_ion = np.fromfile(f, dtype=np.float32)
v_ion = v_ion.reshape(n_ion, 3).T
filename = f"{dirname}/two_stream_electron_v_electron_{step}.bin"
with open(filename, 'rb') as f:
    v_electron = np.fromfile(f, dtype=np.float32)
v_electron = v_electron.reshape(n_electron+n_beam, 3).T

ax.scatter(x_ion[0], v_ion[0]/v_thermal_electron, s=0.1, c='orange', label='ion')
ax.scatter(x_electron[0, :n_electron], v_electron[0, :n_electron]/v_thermal_electron, s=0.1, c='blue', label='electron(background)')
ax.scatter(x_electron[0, n_electron:], v_electron[0, n_electron:]/v_thermal_electron, s=0.1, c='green', marker='*', label='electron(beam)')
ax.set_xlim(0, x_max)
ax.set_ylim(-30, 30)
ax.set_xlabel('$x / \lambda_De$', fontsize=20)
ax.set_ylabel('$V_x / V_{the}$', fontsize=20)
ax.tick_params(labelsize=18)
ax.legend(loc='upper right', fontsize=20)

fig.savefig(savename, dpi=200)


