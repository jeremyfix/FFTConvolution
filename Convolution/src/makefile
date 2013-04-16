INCLUDE_PATH = \
	-I/usr/include \
	-I/usr/local/include \
	-I.
LIBS_PATH = \
	-L/lib \
	-L/usr/lib \
	-L/usr/local/lib

LIBS    = $(LIBS_PATH) -lm 
CFLAGS  = -O3 $(INCLUDE_PATH) 

CFLAS_GSL = \
	`pkg-config --cflags gsl`

CFLAGS_FFTW = \
	`pkg-config --cflags fftw3`

LIBS_GSL = \
	`pkg-config --libs gsl`

LIBS_FFTW = \
	`pkg-config --libs fftw3`

LIBS_CIMG = -lpthread -lX11

COMPILO = g++

BIN_PATH = ../bin

BINS = \
	2DFT.fftw.bin\
	factors.bin\
	std_convolution.bin\
	linear_convolution_gsl_benchmark.gsl.bin\
	circular_convolution_gsl_benchmark.gsl.bin\
	linear_convolution_fftw_benchmark.fftw.bin\
	circular_convolution_fftw_benchmark.fftw.bin\
	fft_fftw.fftw.bin\
	fft_gsl.gsl.bin

all: $(BINS)

%.bin : %.cc
	$(COMPILO) -o $(BIN_PATH)/$@ $(CFLAGS) $< $(LIBS)

%.gsl.bin : %.cc
	$(COMPILO) -o $(BIN_PATH)/$@ $(CFLAGS) $(CFLAGS_GSL) $< $(LIBS) $(LIBS_GSL)

%.fftw.bin : %.cc
	$(COMPILO) -o $(BIN_PATH)/$@ $(CFLAGS) $(CFLAGS_FFTW) $< $(LIBS) $(LIBS_FFTW)

clean : 
	rm -f *.o *~ *.bak $(BIN_PATH)/$(BINS)
