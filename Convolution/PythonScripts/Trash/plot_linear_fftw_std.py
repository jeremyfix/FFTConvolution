import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pylab import *

mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['font.size']=14.0

data = np.loadtxt('benchmarks_LinearConvolution_fftw_all.txt')


fig = plt.figure(figsize=(18,8))
ax = fig.add_subplot(111)

#ax.scatter(data[:,0]**2, data[:,1]**2, data[:,3], c='r')
#ax.scatter(data[:,0]**2, data[:,1]**2, data[:,5], c='y')

l_img_size = data[0,0]
h_img_size = data[np.size(data[:,0])-1,0] - 1
l_kernel_size = data[0,1]
# h_img_size =  this is size dependent and equals N-1

kernel_bounds = h_img_size * np.ones((h_img_size - l_img_size + 1,2))

# For all the image sizes, we determine from which kernel size the FFT based implementation is better
print h_img_size
index  =0
for i in range(l_img_size, h_img_size):
    kernel_bounds[i-l_img_size,0] = i
    for j in range(l_kernel_size, i):
        #print data[index,0], data[index,1], i, j
        if( data[index,3] < 10*data[index,5] and kernel_bounds[i-l_img_size,1] > j):
            kernel_bounds[i-l_img_size,1] = j
        index = index +1

ax.plot(kernel_bounds[:,0] , kernel_bounds[:,1])

ylim([0,100])
#ax.set_xlabel('Image size')
#ax.set_ylabel('Kernel size')
#ax.set_zlabel('Execution time (s.)')
#leg = ax.legend(('Linear FFTW', 'Linear std'),'upper left', shadow=True)

#savefig("benchmark_linear_fftw_std.png")
#savefig("benchmark_linear_fftw_std.pdf")
plt.show()


