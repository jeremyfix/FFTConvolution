import matplotlib as mpl
import numpy as np
from pylab import *

mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['font.size']=14.0

data = np.loadtxt('benchmarks_LinearConvolution_gsl.txt')
data_fftw = np.loadtxt('benchmarks_LinearConvolution_fftw.txt')


fig = figure(figsize=(18,8))
ax = fig.add_subplot(111)

#ax.plot(data[:,3], data_fftw[:,3],'+')

#y_lim = ylim()
#ax.plot([y_lim[0], y_lim[1]], [y_lim[0], y_lim[1]])
#ylim(y_lim)

#ax.plot(data[:,0]**2.0 , data[:,3],'r',data[:,0]**2.0 , data[:,4],'r--',data_fftw[:,0]**2.0 , data_fftw[:,3],'b',data_fftw[:,0]**2.0 , data_fftw[:,4],'b--')

#ax.plot(data[:,0]**2.0 , data[:,3]/data_fftw[:,3],'r',data[:,0]**2.0 , data[:,4]/data_fftw[:,4],'b',data[:,0]**2.0, np.ones(np.size(data[:,0])),'--k')
ax.plot(data[:,0]**2.0 , data[:,3]/data_fftw[:,3],'r')
ax.plot(data[:,0]**2.0 ,np.ones(np.size(data[:,0])),'--k')

xlim([3*3, 512*512])
ylim([0, 2.0])

ax.set_xlabel('Image size (pixels)')
ax.set_ylabel('Execution time (seconds)')
ax.set_title('Ratio GSL/FFTW for performing a 2D linear convolution')
xtickslabel = np.array([])
xtickspositions = np.array([])
for x in data[:,0]:
    if(x%50==0):        
        xtickslabel = np.append(xtickslabel,""+str(int(x))+"x"+str(int(x)))
        xtickspositions = np.append(xtickspositions, x**2.0)
xticks(xtickspositions, xtickslabel)
fig.autofmt_xdate()
#leg = ax.legend(('GSL/FFTW NxN', 'GSL/FFTW 3x3'),'upper left', shadow=True)
#leg = ax.legend(('GSL/FFTW NxN'),'upper left', shadow=True)

savefig("benchmark_linear_convolution_compare.png")
savefig("benchmark_linear_convolution_compare.pdf")
show()


