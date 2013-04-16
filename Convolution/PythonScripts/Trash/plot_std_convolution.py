import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter
from pylab import *

data = np.loadtxt('benchmarks_StdConvolution.txt')


fig = plt.figure()
ax = Axes3D(fig)

#x.scatter(data[:,0]**2, data[:,1]**2, data[:,2], c='r')
#ax.scatter(data[:,0]**2, data[:,1]**2, data[:,3], c='y')

nx = 29
ny = 29
print np.shape(data[0:nx*ny:ny,0])
print np.shape(data[0:ny,1])
print np.shape(np.reshape(data[0:nx*ny,2],(nx, ny)))

X,Y = np.meshgrid(data[0:nx*ny:ny,0],data[0:ny,1])

col = ax.plot_wireframe(X, Y, np.transpose(np.reshape(data[0:nx*ny,2],(nx, ny))))
#col = ax.plot_wireframe(X, Y, 2.0*1e-9 * X * X * Y * Y)
#col.set_color(colorConverter.to_rgba('r'))
col = ax.plot_wireframe(X, Y, np.transpose(np.reshape(data[0:nx*ny,3],(nx, ny))))
col.set_color(colorConverter.to_rgba('r'))

ax.set_xlabel('Image size')
ax.set_ylabel('Kernel size')
ax.set_zlabel('Execution time (s.)')
leg = ax.legend(('Linear', 'Circular'),'upper left', shadow=True)

savefig("benchmark_std_convolution.png")
savefig("benchmark_std_convolution.pdf")
plt.show()


