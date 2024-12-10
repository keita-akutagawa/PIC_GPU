import numpy as np

dx = 0.01
dy = 0.01
x = np.arange(0.0, 4.0 * np.pi, dx)
y = np.arange(-5.0, 5.0, dy)
X, Y = np.meshgrid(x, y)

coefFadeev = 2.0

n = 1.0 / (coefFadeev * np.cos(X) + np.sqrt(1.0 + coefFadeev**2) * np.cosh(Y))**2
n = np.sum(n * dx * dy)

print(n)


