LIBS    = $(LIBS_PATH) -lm `pkg-config --libs fftw3 gsl`
CFLAGS  = -O3 $(INCLUDE_PATH) `pkg-config --cflags fftw3 gsl`

COMPILO = g++
BIN_PATH = ../bin

BINS = \
	2DFT.bin\
	factors.bin\
	fft_fftw.bin\
	fft_gsl.bin\
	std_convolution.bin\
	convolution_gsl_benchmark_linear.bin\
	convolution_gsl_benchmark_linear_optimal.bin\
	convolution_gsl_benchmark_circular.bin\
	convolution_gsl_benchmark_circular_optimal.bin\
	convolution_fftw_benchmark_linear.bin\
	convolution_fftw_benchmark_linear_optimal.bin\
	convolution_fftw_benchmark_circular.bin\
	convolution_fftw_benchmark_circular_optimal.bin\
	convolution_std_benchmark_linear.bin\
	convolution_std_benchmark_circular.bin

all: $(BINS)

%.bin : %.cc
	$(COMPILO) -o $(BIN_PATH)/$@ $(CFLAGS) $< $(LIBS)

convolution_bin:
	g++ -o $(BIN_PATH)/$(TARGET) convolution_benchmark.cc -DVERBOSE=true -DSAVE_RESULTS=true -DCONVOLUTION=$(CONVOLUTION) -DMODE=$(MODE) -O3 -Wall `pkg-config --libs --cflags gsl fftw3`


convolution_fftw_benchmark_linear.bin: convolution_benchmark.cc
	make convolution_bin CONVOLUTION=0 MODE=0 TARGET=$@ 
convolution_fftw_benchmark_linear_optimal.bin: convolution_benchmark.cc
	make convolution_bin CONVOLUTION=0 MODE=1 TARGET=$@ 
convolution_fftw_benchmark_circular.bin: convolution_benchmark.cc
	make convolution_bin CONVOLUTION=0 MODE=2 TARGET=$@ 
convolution_fftw_benchmark_circular_optimal.bin: convolution_benchmark.cc
	make convolution_bin CONVOLUTION=0 MODE=3 TARGET=$@ 


convolution_gsl_benchmark_linear.bin: convolution_benchmark.cc
	make convolution_bin CONVOLUTION=1 MODE=0 TARGET=$@ 
convolution_gsl_benchmark_linear_optimal.bin: convolution_benchmark.cc
	make convolution_bin CONVOLUTION=1 MODE=1 TARGET=$@ 
convolution_gsl_benchmark_circular.bin: convolution_benchmark.cc
	make convolution_bin CONVOLUTION=1 MODE=2 TARGET=$@ 
convolution_gsl_benchmark_circular_optimal.bin: convolution_benchmark.cc
	make convolution_bin CONVOLUTION=1 MODE=3 TARGET=$@ 


convolution_std_benchmark_linear.bin: convolution_benchmark.cc
	make convolution_bin CONVOLUTION=2 MODE=0 TARGET=$@ 
convolution_std_benchmark_circular.bin: convolution_benchmark.cc
	make convolution_bin CONVOLUTION=2 MODE=2 TARGET=$@ 

clean : 
	rm -f *.o *~ *.bak 
	cd $(BIN_PATH) && rm -f $(BINS)
	rm -f ../Data/*.txt
