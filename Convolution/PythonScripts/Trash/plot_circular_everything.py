import matplotlib as mpl
import numpy as np
from pylab import *

# Put arrows for std times larger than the ylim

mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['font.size']=14.0

data = np.loadtxt('benchmarks_circular_everything.txt')
# Contains : img_size kernel_size time_fftw time_gsl time_std

nb_img_sizes = 4
nb_filter_sizes = 7
nb_blanks = 2
x_offset = 2

x_positions = arange(0.0, nb_img_sizes * (nb_filter_sizes+nb_blanks), 1.0)+x_offset
print x_positions
data_gsl = np.zeros(x_positions.shape)
data_fftw = np.zeros(x_positions.shape)
data_std = np.zeros(x_positions.shape)
index = 0
for i in range(nb_img_sizes):
    for j in range(nb_filter_sizes):
        data_fftw[index] = data[i * nb_filter_sizes+j,2] 
        data_gsl[index] = data[i * nb_filter_sizes+j,3]
        data_std[index] = data[i * nb_filter_sizes+j,4]
        index += 1
    data_gsl[index:index+2] = NaN
    data_fftw[index:index+2] = NaN
    data_std[index:index+2] = NaN
    index+=2
    
fig = figure(figsize=(18,8))
ax = fig.add_subplot(111)

ax.plot(x_positions , data_fftw,'k',x_positions , data_gsl,'r',x_positions , data_std,'b')

xlim([0, x_positions.max()])
ylim([0, 1.0])

ax.set_ylabel('Execution time (seconds)')
ax.set_title('Execution time for performing a 2D circular convolution')
leg = ax.legend(('FFTW', 'GSL','Nested for'),'upper left', shadow=True)

# Set up the ticks
xtickslabel = np.array([])
xtickspositions = np.array([])
x_position = 0
for i in range(nb_img_sizes):
    for j in range(nb_filter_sizes):
        xtickslabel = np.append(xtickslabel,""+str(int(data[i * nb_filter_sizes+j,1]))+"x"+str(int(data[i * nb_filter_sizes+j,1])))
        xtickspositions = np.append(xtickspositions, x_position+x_offset)
        x_position += 1
    x_position += 2
xticks(xtickspositions, xtickslabel)
fig.autofmt_xdate()

for i in range(nb_img_sizes):
    text(2+9*i,-0.17,'Image '+str(int(data[i * nb_filter_sizes+j,0]))+"x"+str(int(data[i * nb_filter_sizes+j,0])),fontsize=16)
savefig("benchmark_circular_everything.png")
savefig("benchmark_circular_everything.pdf")
show()


