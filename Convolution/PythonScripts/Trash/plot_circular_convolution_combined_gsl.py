import matplotlib as mpl
import numpy as np
from pylab import *

mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['font.size']=14.0

data = np.loadtxt('benchmarks_CircularConvolution_combined_gsl.txt')

xval = np.arange(0,1500,100)

fig = figure(figsize=(18,8))
ax = fig.add_subplot(111)

#ax.plot(data[:,0]**2.0 , data[:,2],'k', data[:,0]**2.0 , data[:,3],'r',data2[:,0]**2.0 , data2[:,2],'g',data2[:,0]**2.0 , data2[:,3],'y')

ax.plot(data[:,0]**2.0 , data[:,2],'b')

xlim([64*64, 512*512])
ylim([0, 1.0])

ax.set_xlabel('Image size (pixels)')
ax.set_ylabel('Execution time (seconds)')
ax.set_title('Execution time for performing a 2D circular convolution')
xtickslabel = np.array([])
xtickspositions = np.array([])
for x in data[:,0]:
    #if(x != 200 and x!=400 and x!= 1600 and x%100==0):
    if(x%50==0):        
        xtickslabel = np.append(xtickslabel,""+str(int(x))+"x"+str(int(x)))
        xtickspositions = np.append(xtickspositions, x*x)
xticks(xtickspositions, xtickslabel)
fig.autofmt_xdate()
#leg = ax.legend(('GSL NxN'),'upper left', shadow=True)

savefig("benchmark_circular_convolution_combined_gsl.png")
savefig("benchmark_circular_convolution_combined_gsl.pdf")
show()


