import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter
from scipy.optimize import fmin

data = np.genfromtxt('../Data/benchmarks_convolution_fftw.txt', dtype='str')

# Let's filter the different benchmarks
data_linear = np.array(data[np.where(data[:,0] == 'linear'),1:], dtype=np.float)
data_linear = data_linear.reshape(data_linear.shape[1:]) # drop the useless mode dimension
data_linear_optimal = np.array(data[np.where(data[:,0] == 'linear_optimal'),1:], dtype=np.float)
data_linear_optimal = data_linear_optimal.reshape(data_linear_optimal.shape[1:]) # drop the useless mode dimension
data_circular = np.array(data[np.where(data[:,0] == 'circular'),1:], dtype=np.float)
data_circular = data_circular.reshape(data_circular.shape[1:]) # drop the useless mode dimension
data_circular_optimal = np.array(data[np.where(data[:,0] == 'circular_optimal'),1:], dtype=np.float)
data_circular_optimal = data_circular_optimal.reshape(data_circular_optimal.shape[1:]) # drop the useless mode dimension

fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
ax.scatter(data_linear[:,0], data_linear[:,1], data_linear[:,2], c='b')
ax.scatter(data_linear_optimal[:,0], data_linear_optimal[:,1], data_linear_optimal[:,2], c='r')

ax = fig.add_subplot(122, projection='3d')
ax.scatter(data_circular[:,0], data_circular[:,1], data_circular[:,2], c='b')
ax.scatter(data_circular_optimal[:,0], data_circular_optimal[:,1], data_circular_optimal[:,2], c='r')

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



