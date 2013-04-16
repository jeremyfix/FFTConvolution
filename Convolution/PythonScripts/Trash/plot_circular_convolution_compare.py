import matplotlib as mpl
import numpy as np
from pylab import *

mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['font.size']=14.0

data = np.loadtxt('benchmarks_CircularConvolution_gsl.txt')
data_fftw = np.loadtxt('benchmarks_CircularConvolution_fftw.txt')

# Data contains
# image_size kernel_size non_padded padded padded_small_kernel

data_gsl_unpadd = data[:,2]
data_gsl_padd = data[:,3]
diff = data_gsl_unpadd - data_gsl_padd
data_gsl_unpadd[np.where(diff > 0.0)] = 0.0
data_gsl_padd[np.where(diff <= 0.0)] = 0.0
data_gsl = data_gsl_unpadd + data_gsl_padd

fig = figure(figsize=(18,8))
ax = fig.add_subplot(111)


#ax.plot(data[:,0]**2, data_gsl,'r', data_fftw[:,0]**2, data_fftw[:,2],'g')
ax.plot(data[:,0]**2, data_gsl/data_fftw[:,2],'r', data[:,0]**2.0, np.ones(np.size(data[:,0])),'--k')

#y_lim = ylim()
#ax.plot([y_lim[0], y_lim[1]], [y_lim[0], y_lim[1]])
#ylim(y_lim)
#ax.plot(data[:,0]**2.0 , data[:,3],'r',data[:,0]**2.0 , data[:,4],'r--',data_fftw[:,0]**2.0 , data_fftw[:,3],'g',data_fftw[:,0]**2.0 , data_fftw[:,4],'g--')

xlim([0*0, 512*512])
ylim([0, 3.0])

ax.set_xlabel('Image size (pixels)')
ax.set_ylabel('Execution time (seconds)')
ax.set_title('Ratio GSL/FFTW for performing a 2D circular convolution')
xtickslabel = np.array([])
xtickspositions = np.array([])
for x in data[:,0]:
    if(x%50==0):        
        xtickslabel = np.append(xtickslabel,""+str(int(x))+"x"+str(int(x)))
        xtickspositions = np.append(xtickspositions, x**2.0)
xticks(xtickspositions, xtickslabel)
fig.autofmt_xdate()
#leg = ax.legend(('GSL/FFTW NxN'),'upper left', shadow=True)

savefig("benchmark_circular_convolution_compare.png")
savefig("benchmark_circular_convolution_compare.pdf")
show()


