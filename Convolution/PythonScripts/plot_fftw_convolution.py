import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter
from scipy.optimize import fmin
from matplotlib import cm, colors
import copy


# Utilitary functions to adjust the colormaps
# from : https://sites.google.com/site/theodoregoetz/notes/matplotlib_colormapadjust
def cmap_powerlaw_adjust(cmap, a):
    '''
    returns a new colormap based on the one given
    but adjusted via power-law:

    newcmap = oldcmap**a
    '''
    if a < 0.:
        return cmap
    cdict = copy.copy(cmap._segmentdata)
    fn = lambda x : (x[0]**a, x[1], x[2])
    for key in ('red','green','blue'):
        cdict[key] = map(fn, cdict[key])
        cdict[key].sort()
        assert (cdict[key][0]<0 or cdict[key][-1]>1), \
            "Resulting indices extend out of the [0, 1] segment."
    return colors.LinearSegmentedColormap('colormap',cdict,1024)

def cmap_center_adjust(cmap, center_ratio):
    '''
    returns a new colormap based on the one given
    but adjusted so that the old center point higher
    (>0.5) or lower (<0.5)
    '''
    if not (0. < center_ratio) & (center_ratio < 1.):
        return cmap
    a = np.log(center_ratio) / np.log(0.5)
    return cmap_powerlaw_adjust(cmap, a)

def cmap_center_point_adjust(cmap, range, center):
    '''
    converts center to a ratio between 0 and 1 of the
    range given and calls cmap_center_adjust(). returns
    a new adjusted colormap accordingly
    '''
    if not ((range[0] < center) and (center < range[1])):
        return cmap
    return cmap_center_adjust(cmap,
        abs(center - range[0]) / abs(range[1] - range[0]))



data = np.genfromtxt('../Data/benchmarks_convolution_fftw.txt', dtype='str')

# Let's filter the different benchmarks
data_linear = np.array(data[np.where(data[:,0] == 'linear'),1:], dtype=np.float)
data_linear = data_linear.reshape(data_linear.shape[1:]) # drop the useless mode dimension
data_linear_unpadded = np.array(data[np.where(data[:,0] == 'linear_unpadded'),1:], dtype=np.float)
data_linear_unpadded = data_linear_unpadded.reshape(data_linear_unpadded.shape[1:]) # drop the useless mode dimension
data_circular = np.array(data[np.where(data[:,0] == 'circular'),1:], dtype=np.float)
data_circular = data_circular.reshape(data_circular.shape[1:]) # drop the useless mode dimension
data_circular_unpadded = np.array(data[np.where(data[:,0] == 'circular_unpadded'),1:], dtype=np.float)
data_circular_unpadded = data_circular_unpadded.reshape(data_circular_unpadded.shape[1:]) # drop the useless mode dimension

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

Z_linear_ratio = Z_linear_unpadded / Z_linear 

Z_circular = np.NaN * np.zeros(X.shape)
Z_circular_unpadded = np.NaN * np.zeros(X.shape)
for d in data_circular:
    Z_circular[d[1]-min_kernel, d[0]-min_src] = d[2]
for d in data_circular_unpadded:
    Z_circular_unpadded[d[1]-min_kernel, d[0]-min_src] = d[2]

Z_circular_ratio = Z_circular_unpadded / Z_circular

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
ax.plot_wireframe(X, Y, Z_circular_unpadded, color='r')
ax.set_xlabel('Source size (N)')
ax.set_ylabel('Kernel size (k)')
ax.set_zlabel('Time (s.)')
ax.set_title('Circular convolution of a source NxN with a kernel kxk')



######################################## Get a fit of the performances
# Fit the execution times on the optimal convolutions, they are smoother

# Temps : Nlog(N)

def f_lin(x):
    return np.sum(((x[0] * (data_linear[:,0]+data_linear[:,1]/2.0)**2 * np.log(data_linear[:,0]+data_linear[:,1]/2.0)) - data_linear[:,2])**2)
xopt_lin = fmin(f_lin, [np.random.random()*1e-7], xtol=1e-10)
print xopt_lin

def f_circ(x):
    return np.sum(((x[0] * (data_circular[:,0]+data_circular[:,1])**2 * np.log(data_circular[:,0]+data_circular[:,1])) - data_circular[:,2])**2)
xopt_circ = fmin(f_circ, [np.random.random()*1e-7], xtol=1e-8)
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


# Plot the ratio of the padded/unpadded convolutions

cmap = cm.seismic

extent = [np.min(X),np.max(X),np.min(Y),np.max(Y)]

plotkwargs = {
    'extent' : extent,
    'origin' : 'lower',
    'interpolation' : 'none',
    'aspect' : 'auto'}

fig3 = plt.figure(figsize=(8,3))
fig3.subplots_adjust(left=.05,bottom=.11,right=.94,top=.83,wspace=.35)
ax = fig3.add_subplot(121)
im = ax.imshow(Z_linear_ratio,
                   cmap=cmap_center_point_adjust(cmap,[0, 4],1.0),
                   **plotkwargs)
im.set_clim([0,4])
cb = ax.figure.colorbar(im, ax=ax)
ax.set_title('Linear convolution ratio (unpadded / padded)')

ax = fig3.add_subplot(122)
im = ax.imshow(Z_circular_ratio,
                   cmap=cmap_center_point_adjust(cmap, [0, 4],1.0),
                   **plotkwargs)
im.set_clim([0,4])
cb = ax.figure.colorbar(im, ax=ax)
ax.set_title('Circular convolution ratio (padded / unpadded)')

plt.show()
