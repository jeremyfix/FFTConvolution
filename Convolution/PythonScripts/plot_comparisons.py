# -*- coding: utf-8 -*-

# In this script we plot surface of ratio of running time
# for the different convolutions, linear_same and circular_full, 
# and the different libraries :
# STD
# Octave
# FFTW
# GSL
# The resulting plot is a grid of 4 x 4 subplots where we plot the time
# of each library against each other

import numpy as np
import matplotlib.pyplot as plt
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
data_linear_fftw = np.array(data[np.where(data[:,0] == 'linear'),1:], dtype=np.float)
data_linear_fftw = data_linear_fftw.reshape(data_linear_fftw.shape[1:]) # drop the useless mode dimension
data_circular_fftw = np.array(data[np.where(data[:,0] == 'circular'),1:], dtype=np.float)
data_circular_fftw = data_circular_fftw.reshape(data_circular_fftw.shape[1:]) # drop the useless mode dimension

data = np.genfromtxt('../Data/benchmarks_convolution_gsl.txt', dtype='str')
data_linear_gsl = np.array(data[np.where(data[:,0] == 'linear'),1:], dtype=np.float)
data_linear_gsl = data_linear_gsl.reshape(data_linear_gsl.shape[1:]) # drop the useless mode dimension
data_circular_gsl = np.array(data[np.where(data[:,0] == 'circular'),1:], dtype=np.float)
data_circular_gsl = data_circular_gsl.reshape(data_circular_gsl.shape[1:]) # drop the useless mode dimension

data = np.genfromtxt('../Data/benchmarks_convolution_std.txt', dtype='str')
data_linear_std = np.array(data[np.where(data[:,0] == 'linear'),1:], dtype=np.float)
data_linear_std = data_linear_std.reshape(data_linear_std.shape[1:]) # drop the useless mode dimension
data_circular_std = np.array(data[np.where(data[:,0] == 'circular'),1:], dtype=np.float)
data_circular_std = data_circular_std.reshape(data_circular_std.shape[1:]) # drop the useless mode dimension

data = np.genfromtxt('../Data/benchmarks_convolution_octave.txt', dtype='str')
data_linear_octave = np.array(data[np.where(data[:,0] == 'linear'),1:], dtype=np.float)
data_linear_octave = data_linear_octave.reshape(data_linear_octave.shape[1:]) # drop the useless mode dimension
data_circular_octave = np.array(data[np.where(data[:,0] == 'circular'),1:], dtype=np.float)
data_circular_octave = data_circular_octave.reshape(data_circular_octave.shape[1:]) # drop the useless mode dimension

# We suppose that both the linear and circular convolutions were using the same source and filter sizes
min_src = data_linear_fftw[:,0].min()
max_src = data_linear_fftw[:,0].max()
min_kernel = data_linear_fftw[:,1].min()
max_kernel = data_linear_fftw[:,1].max()
X,Y = np.meshgrid(np.arange(min_src, max_src+1),
                  np.arange(min_kernel, max_kernel+1))

Z_linear_fftw = np.NaN * np.zeros(X.shape)
for d in data_linear_fftw:
    Z_linear_fftw[d[1]-min_kernel, d[0]-min_src] = d[2]
Z_circular_fftw = np.NaN * np.zeros(X.shape)
for d in data_circular_fftw:
    Z_circular_fftw[d[1]-min_kernel, d[0]-min_src] = d[2]

Z_linear_gsl = np.NaN * np.zeros(X.shape)
for d in data_linear_gsl:
    Z_linear_gsl[d[1]-min_kernel, d[0]-min_src] = d[2]
Z_circular_gsl = np.NaN * np.zeros(X.shape)
for d in data_circular_gsl:
    Z_circular_gsl[d[1]-min_kernel, d[0]-min_src] = d[2]


Z_linear_std = np.NaN * np.zeros(X.shape)
for d in data_linear_std:
    Z_linear_std[d[1]-min_kernel, d[0]-min_src] = d[2]
Z_circular_std = np.NaN * np.zeros(X.shape)
for d in data_circular_std:
    Z_circular_std[d[1]-min_kernel, d[0]-min_src] = d[2]

Z_linear_octave = np.NaN * np.zeros(X.shape)
for d in data_linear_octave:
    Z_linear_octave[d[1]-min_kernel, d[0]-min_src] = d[2]
Z_circular_octave = np.NaN * np.zeros(X.shape)
for d in data_circular_octave:
    Z_circular_octave[d[1]-min_kernel, d[0]-min_src] = d[2]


Z_names = ['fftw', 'gsl', 'std', 'octave']
Z_linear = [Z_linear_fftw, Z_linear_gsl, Z_linear_std, Z_linear_octave]
Z_circular = [Z_circular_fftw, Z_circular_gsl, Z_circular_std, Z_circular_octave]

###################################################################
# Plot the ratio of linear convolutions
cmap = cm.seismic
extent = [np.min(X),np.max(X),np.min(Y),np.max(Y)]

plotkwargs = {
    'extent' : extent,
    'origin' : 'lower',
    'interpolation' : 'none',
    'aspect' : 'equal'}

fig = plt.figure(figsize=(10,10), facecolor='w')
fig.subplots_adjust(left=.05,bottom=.11,right=.94,top=.83,wspace=.35)

for i, zi in enumerate(Z_linear):
    for j, zj in enumerate(Z_linear):
        # Compute the ratio zi / zj
        z_ratio = zi / zj
        ax = fig.add_subplot(len(Z_linear),len(Z_linear),i*len(Z_linear) + j + 1)
        im = ax.imshow(z_ratio,
                       cmap=cmap_center_point_adjust(cmap,[0, 10],1.0),
                       **plotkwargs)
        im.set_clim([0,10])
        cb = ax.figure.colorbar(im, ax=ax)
        ax.set_title('Ratio %s/%s' % (Z_names[i],Z_names[j]))
        ax.set_xlabel('Source size')
        ax.set_ylabel('Kernel size')

plt.savefig('comparison_linear.png', bbox_inches='tight')

###################################################################
# Plot the ratio of circular convolutions
cmap = cm.seismic
extent = [np.min(X),np.max(X),np.min(Y),np.max(Y)]

plotkwargs = {
    'extent' : extent,
    'origin' : 'lower',
    'interpolation' : 'none',
    'aspect' : 'equal'}

fig = plt.figure(figsize=(10,10), facecolor='w')
fig.subplots_adjust(left=.05,bottom=.11,right=.94,top=.83,wspace=.35)

for i, zi in enumerate(Z_circular):
    for j, zj in enumerate(Z_circular):
        # Compute the ratio zi / zj
        z_ratio = zi / zj
        ax = fig.add_subplot(len(Z_circular),len(Z_circular),i*len(Z_circular) + j + 1)
        #ax = axs[i][j]
        im = ax.imshow(z_ratio,
                       cmap=cmap_center_point_adjust(cmap,[0, 10],1.0),
                       **plotkwargs)
        im.set_clim([0,10])
        cb = ax.figure.colorbar(im, ax=ax)
        ax.set_title('Ratio %s/%s' % (Z_names[i],Z_names[j]))
        ax.set_xlabel('Source size')
        ax.set_ylabel('Kernel size')

plt.savefig('comparison_circular.png', bbox_inches='tight')

plt.show()
