import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter
from scipy.optimize import fmin

data = np.genfromtxt('../Data/benchmarks_convolution_std.txt', dtype='str')

# Let's filter the different benchmarks
data_linear = np.array(data[np.where(data[:,0] == 'linear'),1:], dtype=np.float)
data_linear = data_linear.reshape(data_linear.shape[1:]) # drop the useless mode dimension

data_circular = np.array(data[np.where(data[:,0] == 'circular'),1:], dtype=np.float)
data_circular = data_circular.reshape(data_circular.shape[1:]) # drop the useless mode dimension

# fig = plt.figure()
# ax = fig.add_subplot(121, projection='3d')
# ax.scatter(data_linear[:,0], data_linear[:,1], data_linear[:,2], c='b')

# ax = fig.add_subplot(122, projection='3d')
# ax.scatter(data_circular[:,0], data_circular[:,1], data_circular[:,2], c='b')

# plt.show()

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
ax = fig.add_subplot(121, projection='3d')
ax.plot_wireframe(X, Y, Z_linear)
ax.set_xlabel('Source size (N)')
ax.set_ylabel('Kernel size (k)')
ax.set_zlabel('Time (s.)')
ax.set_title('Linear convolution of a source NxN with a kernel kxk')

ax = fig.add_subplot(122, projection='3d')
ax.plot_wireframe(X, Y, Z_circular)
ax.set_xlabel('Source size (N)')
ax.set_ylabel('Kernel size (k)')
ax.set_zlabel('Time (s.)')
ax.set_title('Circular convolution of a source NxN with a kernel kxk')

plt.show()

######################################## Get a fit of the performances
# Fit the execution times

# def f_lin(x):
#     global data
#     return np.sum((x[0] * (data[:,0]**2 * data[:,1]**2 -data[:,1]**2/4.0 + data[:,1]/4.0)- data[:,2])**2)
# xopt_lin = fmin(f_lin, [1e-8], xtol=1e-10)
# print xopt_lin

# def f_circ(x):
#     global data
#     return np.sum((x[0] * data[:,0]**x[1] * data[:,1]**x[2] - data[:,3])**2)
# xopt_circ = fmin(f_circ, [1e-9,2.0,2.0], xtol=1e-8)
# print xopt_circ
######################################## 



