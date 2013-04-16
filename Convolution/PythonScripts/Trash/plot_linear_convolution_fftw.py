import matplotlib as mpl
import numpy as np
from pylab import *

mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['font.size']=14.0

data = np.loadtxt('benchmarks_LinearConvolution_fftw.txt')

# Check the errors
errorbound  = 1e-10
if(np.sum(np.where(data[:,6]>errorbound)) != 0):
    print "Errors for sizes : ", data[np.where(data[:,6]>errorbound),1:2]
if(np.sum(np.where(data[:,7]>errorbound)) != 0):
    print "Errors for sizes : ", data[np.where(data[:,7]>errorbound),1:2]

fig = figure(figsize=(18,8))
ax = fig.add_subplot(111)

ax.plot(data[:,0]**2.0 , data[:,2],'b', data[:,0]**2.0 , data[:,3],'r',data[:,0]**2.0 , data[:,4],'r--')

xlim([3**2, 512**2])
ylim([0, 1.0])

ax.set_xlabel('Image size (pixels)')
ax.set_ylabel('Execution time (seconds)')
ax.set_title('Execution time for performing a 2D linear convolution')
xtickslabel = np.array([])
xtickspositions = np.array([])
for x in data[:,0]:
    if(x%50==0 and x <= 450):        
        xtickslabel = np.append(xtickslabel,""+str(int(x))+"x"+str(int(x)))
        xtickspositions = np.append(xtickspositions, x**2)
xticks(xtickspositions, xtickslabel)
fig.autofmt_xdate()
leg = ax.legend(('FFTW NxN', 'FFTW Optimal NxN','FFTW Optimal 3x3'),'upper left', shadow=True)

savefig("benchmark_linear_convolution_fftw.png")
savefig("benchmark_linear_convolution_fftw.pdf")
show()


