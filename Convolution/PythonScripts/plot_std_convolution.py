import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter
from pylab import *
from scipy.optimize import fmin

data = np.loadtxt('../Data/benchmarks_convolution_std.txt')

# Fit the execution times
#def f_lin(x):
#    global data
#    return np.sum((x[0] * data[:,0]**x[1] * data[:,1]**x[2] - data[:,2])**2)
#xopt_lin = fmin(f_lin, [1e-9, 2.0,2.0], xtol=1e-8)
#print xopt_lin

def f_lin(x):
    global data
    return np.sum((x[0] * (data[:,0]**2 * data[:,1]**2 -data[:,1]**2/4.0 + data[:,1]/4.0)- data[:,2])**2)
xopt_lin = fmin(f_lin, [1e-8], xtol=1e-10)
print xopt_lin

def f_circ(x):
    global data
    return np.sum((x[0] * data[:,0]**x[1] * data[:,1]**x[2] - data[:,3])**2)
xopt_circ = fmin(f_circ, [1e-9,2.0,2.0], xtol=1e-8)
print xopt_circ

fig = plt.figure()
ax = Axes3D(fig)

nx = 27
ny = 62

X,Y = np.meshgrid(data[0:nx*ny:ny,0],data[0:ny,1])

col = ax.plot_wireframe(X, Y, np.transpose(np.reshape(data[0:nx*ny,3],(nx, ny))))
col.set_color(colorConverter.to_rgba('r'))
#col = ax.plot_wireframe(X, Y, xopt_circ[0] * X**xopt_circ[1] * Y**xopt_circ[2])
#col.set_color(colorConverter.to_rgba('k'))
col = ax.plot_wireframe(X, Y, np.transpose(np.reshape(data[0:nx*ny,2],(nx, ny))))
col.set_color(colorConverter.to_rgba('b'))
#col = ax.plot_wireframe(X, Y, xopt_lin[0] * (X**xopt_lin[1])*(Y**xopt_lin[2]))
col = ax.plot_wireframe(X, Y, xopt_lin[0] * (X**2 * Y**2 -Y**2/4.0 + Y/4.0))
col.set_color(colorConverter.to_rgba('y'))


ax.set_xlabel('Image size')
ax.set_ylabel('Kernel size')
ax.set_zlabel('Execution time (s.)')
leg = ax.legend(('Circular', 'Linear'),'upper left', shadow=True)

savefig("../Images/benchmark_std_convolution.png")
savefig("../Images/benchmark_std_convolution.pdf")
plt.show()


