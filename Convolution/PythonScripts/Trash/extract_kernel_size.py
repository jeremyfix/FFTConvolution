import numpy as np
import matplotlib as mpl
from pylab import *

data_std = np.loadtxt('benchmarks_StdConvolution.txt')
data_fftw = np.loadtxt('benchmarks_LinearConvolution_fftw.txt')
print "ok", data_fftw.shape
n_kernels = 29
n_images = int(np.shape(data_std)[0]/n_kernels)-1
print n_images
data_linear = np.zeros((n_images,2))

print "Linear Convolution"
file_linear = open('kernel_sizes_linear.txt','w')
for i in range(n_images):
    print i, data_fftw[i,0], data_fftw[i,3]
    t_fftw = data_fftw[i,3]
    j = 0
    while( j < n_kernels):
        if(data_std[i*n_kernels+j,2] > t_fftw):
            break
        j+=1
    #print data_fftw[i,0],data_std[i*n_kernels+j,1]
    file_linear.write(""+str(data_fftw[i,0])+"\t"+str(data_std[i*n_kernels+j,1])+'\n')
    data_linear[i,0] = data_fftw[i,0]
    data_linear[i,1] = data_std[i*n_kernels+j,1]
file_linear.close()

data_std = np.loadtxt('benchmarks_StdConvolution.txt')
data_fftw = np.loadtxt('benchmarks_CircularConvolution_fftw.txt')
data_circular = np.zeros((n_images,2))

print "Circular Convolution"
file_circular = open('kernel_sizes_circular.txt','w')
for i in range(n_images):
    t_fftw = data_fftw[i,2]
    j = 0
    while( j < n_kernels):
        if(data_std[i*n_kernels+j,3] > t_fftw):
            break
        j+=1
    #print data_fftw[i,0],data_std[i*n_kernels+j,1]
    file_circular.write(""+str(data_fftw[i,0])+"\t"+str(data_std[i*n_kernels+j,1])+'\n')
    data_circular[i,0] = data_fftw[i,0]
    data_circular[i,1] = data_std[i*n_kernels+j,1]
file_circular.close()


mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['font.size']=14.0

fig = figure(figsize=(18,8))
ax = fig.add_subplot(111)

ax.plot(data_linear[:,0]**2 , data_linear[:,1]**2,'b', data_circular[:,0]**2.0 , data_circular[:,1]**2,'r')

xlim([64*64, 512*512])
ylim([0, n_kernels*n_kernels])

ax.set_xlabel('Image size (pixels)')
ax.set_ylabel('Kernel size (pixels)')
ax.set_title('Kernel size for which a FFTW convolution is always faster than a standard convolution')
xtickslabel = np.array([])
xtickspositions = np.array([])
for x in data_linear[:,0]:
    if(x%50==0):        
        xtickslabel = np.append(xtickslabel,""+str(int(x))+"x"+str(int(x)))
        xtickspositions = np.append(xtickspositions, x**2.0)
xticks(xtickspositions, xtickslabel)
ytickslabel = np.array([])
ytickspositions = np.array([])
for x in data_std[0:n_kernels,1]:
    if(x%2 == 0):
        ytickslabel =  np.append(ytickslabel,""+str(int(x))+"x"+str(int(x)))
        ytickspositions = np.append(ytickspositions, x**2.0)

yticks(ytickspositions, ytickslabel)
fig.autofmt_xdate()
#fig.autofmt_ydate()
leg = ax.legend(('Linear', 'Circular'),'upper left', shadow=True)

savefig("kernel_size.png")
savefig("kernel_size.pdf")
show()
