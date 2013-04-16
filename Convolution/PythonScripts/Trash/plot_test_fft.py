import matplotlib as mpl
import numpy as np
from pylab import *

mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['font.size']=14.0

data = np.loadtxt('benchmarks_TestFFT.txt')

xval = np.arange(0,1500,100)

fig = figure(figsize=(15,8))
ax = fig.add_subplot(111)

print np.sum(np.where(data[:,4] > 1e-10)), " FFT mistakes ?"

ax.plot(data[:,0] , data[:,2],'b', data[:,0] , data[:,3],'g', data[:,0], 2e-8 * data[:,0] **3,'r--', data[:,0], 5e-8 * data[:,0]*data[:,0]*np.log(data[:,0]),'r')

#5e-10 * data[:,0]*data[:,0]*data[:,0]

xlim([0, 300])
ylim([0, 0.2])

ax.set_xlabel('Image size (pixels)')
ax.set_ylabel('Execution time (seconds)')
ax.set_title('Execution time for computing a 2D FFT')
xtickslabel = np.array([])
xtickspositions = np.array([])
for x in data[:,0]:
    if(x%50==0):        
        xtickslabel = np.append(xtickslabel,""+str(int(x))+"x"+str(int(x)))
        xtickspositions = np.append(xtickspositions, x)
xticks(xtickspositions, xtickslabel)        
fig.autofmt_xdate()        
leg = ax.legend(('GSL', 'FFTW', '2e-8 N^3', '5e-8 N^2log(N)'),
           'upper left', shadow=True)

savefig("test_fft.png")
savefig("test_fft.pdf")
show()


