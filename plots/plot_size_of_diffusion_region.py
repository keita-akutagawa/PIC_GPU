import numpy as np 
import matplotlib.pyplot as plt 
import os 


def apply_convolution(data, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
    return np.convolve(data.flatten(), kernel.flatten(), mode='same').reshape(data.shape)


fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(111)

savename = "De_size.png"

De_mr1 = np.load("1400_mr1_De.npy")
De_mr2 = np.load("3000_mr2_De.npy")
De_mr4 = np.load("8000_mr4_De.npy")
De_mr8 = np.load("20000_mr8_De.npy")
De_mr16 = np.load("48000_mr16_De.npy")
De_mr32 = np.load("108800_mr32_De.npy")

for i, De in enumerate([De_mr1, De_mr2, De_mr4, De_mr8, De_mr16, De_mr32]):

    c = 1.0
    epsilon0 = 1.0
    mu_0 = 1.0 / (epsilon0 * c**2)
    m_unit = 1.0
    r_m = 1 / 2**i
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
    electron_inertial_length = c / omega_pe
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

    local_min = int(40 * ion_inertial_length)
    local_max = int(60 * ion_inertial_length)

    De = apply_convolution(De, int(electron_inertial_length))
    
    y_coordinate = np.arange(De.shape[0]) * dy
    ax1.plot(
        x_coordinate[local_min:local_max] / ion_inertial_length, 
        De[n_y//2, local_min:local_max] / np.max(De[n_y//2, local_min:local_max]), 
        label=f"mass ratio = {int(2**i)}"
    )


ax1.grid()
ax1.set_xlabel(r'$x / \lambda_i$', fontsize=20)
ax1.set_ylabel(r'$D_e / D_{e, max}$', fontsize=20)
ax1.set_xlim(40, 60)
ax1.set_ylim(0.0, 1.2)
ax1.tick_params(labelsize=18)
ax1.legend(loc='upper right', fontsize=20)


plt.tight_layout()
fig.savefig(savename, dpi=200)


