import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter
from scipy.optimize import fmin

data = np.genfromtxt('../Data/benchmarks_convolution_gsl.txt', dtype='str')

# Let's filter the different benchmarks
data_linear = np.array(data[np.where(data[:,0] == 'linear'),1:], dtype=np.float)
data_linear = data_linear.reshape(data_linear.shape[1:]) # drop the useless mode dimension
data_linear_unpadded = np.array(data[np.where(data[:,0] == 'linear_unpadded'),1:], dtype=np.float)
data_linear_unpadded = data_linear_unpadded.reshape(data_linear_unpadded.shape[1:]) # drop the useless mode dimension
data_circular = np.array(data[np.where(data[:,0] == 'circular'),1:], dtype=np.float)
data_circular = data_circular.reshape(data_circular.shape[1:]) # drop the useless mode dimension
data_circular_padded = np.array(data[np.where(data[:,0] == 'circular_padded'),1:], dtype=np.float)
data_circular_padded = data_circular_padded.reshape(data_circular_padded.shape[1:]) # drop the useless mode dimension

min_src = data_linear[:,0].min()
max_src = data_linear[:,0].max()
min_kernel = data_linear[:,1].min()
max_kernel = data_linear[:,1].max()

X,Y = np.meshgrid(np.arange(min_src, max_src+1),
                  np.arange(min_kernel, max_kernel+1))
Z_linear = np.NaN * np.zeros(X.shape)
Z_linear_unpadded = np.NaN * np.zeros(X.shape)
for d in data_linear:
    Z_linear[d[1]-min_kernel, d[0]-min_src] = d[2]
for d in data_linear_unpadded:
    Z_linear_unpadded[d[1]-min_kernel, d[0]-min_src] = d[2]

Z_circular = np.NaN * np.zeros(X.shape)
Z_circular_padded = np.NaN * np.zeros(X.shape)
for d in data_circular:
    Z_circular[d[1]-min_kernel, d[0]-min_src] = d[2]
for d in data_circular_padded:
    Z_circular_padded[d[1]-min_kernel, d[0]-min_src] = d[2]

fig =plt.figure(figsize=(15,5))
ax = fig.add_subplot(121, projection='3d')
ax.plot_wireframe(X, Y, Z_linear, color='b', alpha=0.4)
ax.plot_wireframe(X, Y, Z_linear_unpadded, color='r')
ax.set_xlabel('Source size (N)')
ax.set_ylabel('Kernel size (k)')
ax.set_zlabel('Time (s.)')
ax.set_title('Linear convolution of a source NxN with a kernel kxk')

ax = fig.add_subplot(122, projection='3d')
ax.plot_wireframe(X, Y, Z_circular, color='b', alpha=0.4)
ax.plot_wireframe(X, Y, Z_circular_padded, color='r')
ax.set_xlabel('Source size (N)')
ax.set_ylabel('Kernel size (k)')
ax.set_zlabel('Time (s.)')
ax.set_title('Circular convolution of a source NxN with a kernel kxk')



######################################## Get a fit of the performances
# Fit the execution times on the optimal convolutions, they are smoother

# Temps : Nlog(N)

def f_lin(x):
    return np.sum(((x[0] * (data_linear[:,0]+data_linear[:,1]/2.0)**2 * np.log(data_linear[:,0]+data_linear[:,1]/2.0)) - data_linear[:,2])**2)
xopt_lin = fmin(f_lin, [1e-7], xtol=1e-10)
print xopt_lin

def f_circ(x):
    return np.sum(((x[0] * (data_circular[:,0]+data_circular[:,1])**2 * np.log(data_circular[:,0]+data_circular[:,1])) - data_circular[:,2])**2)
xopt_circ = fmin(f_circ, [1e-7], xtol=1e-8)
print xopt_circ
######################################## 

Z_linear_fit = xopt_lin[0] * ((X+Y/2.0)**2 * np.log(X+Y/2.0))
Z_circular_fit = xopt_circ[0] * ((X+Y)**2 * np.log(X+Y))

fig2 = plt.figure()
ax1 = fig2.add_subplot(121, projection='3d')
ax1.plot_wireframe(X, Y, Z_linear, color='r')
ax1.plot_wireframe(X, Y, Z_linear_fit, color='b', alpha=0.1)

ax1.set_xlabel('Source size (N)')
ax1.set_ylabel('Kernel size (k)')
ax1.set_zlabel('Time (s.)')
ax1.set_title('Linear convolution of a source NxN with a kernel kxk')


ax2 = fig2.add_subplot(122, projection='3d')
ax2.plot_wireframe(X, Y, Z_circular, color='r')
ax2.plot_wireframe(X, Y, Z_circular_fit, color='b', alpha=0.1)

ax2.set_xlabel('Source size (N)')
ax2.set_ylabel('Kernel size (k)')
ax2.set_zlabel('Time (s.)')
ax2.set_title('Circular convolution of a source NxN with a kernel kxk')

plt.show()
