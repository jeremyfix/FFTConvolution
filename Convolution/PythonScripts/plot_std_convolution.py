import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter
from scipy.optimize import fmin

# Results found by the fit :
# Linear , Time = 1.37500000e-09 * N^2 * k^2
# Circular , Time = 3.18750000e-09 * N^2 * k^2

data = np.genfromtxt('../Data/benchmarks_convolution_std.txt', dtype='str')

# Let's filter the different benchmarks
data_linear = np.array(data[np.where(data[:,0] == 'linear'),1:], dtype=np.float)
data_linear = data_linear.reshape(data_linear.shape[1:]) # drop the useless mode dimension

data_circular = np.array(data[np.where(data[:,0] == 'circular'),1:], dtype=np.float)
data_circular = data_circular.reshape(data_circular.shape[1:]) # drop the useless mode dimension

min_src = data_linear[:,0].min()
max_src = data_linear[:,0].max()
min_kernel = data_linear[:,1].min()
max_kernel = data_linear[:,1].max()

X,Y = np.meshgrid(np.arange(min_src, max_src+1),
                  np.arange(min_kernel, max_kernel+1))
Z_linear = np.NaN * np.zeros(X.shape)
for d in data_linear:
    Z_linear[d[1]-min_kernel, d[0]-min_src] = d[2]
Z_circular = np.NaN * np.zeros(X.shape)
for d in data_circular:
    Z_circular[d[1]-min_kernel, d[0]-min_src] = d[2]

fig =plt.figure()
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_wireframe(X, Y, Z_linear)
ax1.set_xlabel('Source size (N)')
ax1.set_ylabel('Kernel size (k)')
ax1.set_zlabel('Time (s.)')
ax1.set_title('Linear convolution of a source NxN with a kernel kxk')

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_wireframe(X, Y, Z_circular)
ax2.set_xlabel('Source size (N)')
ax2.set_ylabel('Kernel size (k)')
ax2.set_zlabel('Time (s.)')
ax2.set_title('Circular convolution of a source NxN with a kernel kxk')



######################################## Get a fit of the performances
# Fit the execution times

def f_lin(x):
    return np.sum( (x[0] * data_linear[:,0]**2 * data_linear[:,1]**2 - data_linear[:,2])**2)
xopt_lin = fmin(f_lin, [1e-8], xtol=1e-10)
print xopt_lin

def f_circ(x):
    return np.sum( (x[0] * data_circular[:,0]**2 * data_circular[:,1]**2 - data_circular[:,2])**2)
xopt_circ = fmin(f_circ, [1e-8], xtol=1e-10)
print xopt_circ

Z_linear_fit = xopt_lin[0] * X** 2 * Y**2
ax1.plot_wireframe(X, Y, Z_linear_fit, alpha=0.2, color='r')

Z_circular_fit = xopt_circ[0] * X** 2 * Y**2
ax2.plot_wireframe(X, Y, Z_circular_fit, alpha=0.2, color='r')
plt.show()

######################################## 




