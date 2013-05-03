LIBS    = $(LIBS_PATH) -lm `pkg-config --libs fftw3 gsl`
CFLAGS  = -O3 $(INCLUDE_PATH) `pkg-config --cflags fftw3 gsl`

COMPILO = g++

BINS_BENCHMARKS=\
	convolution_fftw_benchmark_linear.bin\
	convolution_fftw_benchmark_linear_unpadded.bin\
	convolution_fftw_benchmark_circular.bin\
	convolution_fftw_benchmark_circular_unpadded.bin\
	convolution_gsl_benchmark_linear.bin\
	convolution_gsl_benchmark_linear_unpadded.bin\
	convolution_gsl_benchmark_circular.bin\
	convolution_gsl_benchmark_circular_unpadded.bin\
	convolution_octave_benchmark_linear.bin\
	convolution_octave_benchmark_circular.bin\
	convolution_std_benchmark_linear.bin\
	convolution_std_benchmark_circular.bin

BINS_BENCHMARKS_SCRIPTS=\
	run_gsl_bench.sh \
	run_fftw_bench.sh \
	run_std_bench.sh \
	run_octave_bench.sh

MAX_SIZE_BENCHMARK= 100


BINS_TEST=\
	convolution_test_gsl.bin\
	convolution_test_fftw.bin\
	convolution_test_std.bin

BINS = \
	2DFT.bin\
	factors.bin\
	fft_fftw.bin\
	fft_gsl.bin\
	$(BINS_TEST)\
	$(BINS_BENCHMARKS)



all: 
	@echo "Targets :" 
	@echo " make benchmarks : executes the benchmarks of the different convolution implementations"
	@echo " make test : run the comparisons of our libs against octave and matlab"
	@echo " make bins : compile all the codes"


bins:$(BINS)

benchmarks:$(BINS_BENCHMARKS_SCRIPTS) $(BINS_BENCHMARKS)
	@for bin_script in $(BINS_BENCHMARKS_SCRIPTS); do \
		sh $$bin_script $(MAX_SIZE_BENCHMARK); \
	done

test:$(BINS_TEST)
	@for bin_script in $(BINS_TEST); do \
		./$$bin_script; \
	done

%.bin : %.cc
	$(COMPILO) -o $@ $(CFLAGS) $< $(LIBS)

convolution_bin:
	g++ -o $(TARGET) convolution_benchmark.cc -DVERBOSE=true -DSAVE_RESULTS=true -DCONVOLUTION=$(CONVOLUTION) -DMODE=$(MODE) -O3 -Wall $(CFLAGSLIBS)


convolution_fftw_benchmark_linear.bin: convolution_benchmark.cc convolution_fftw.h
	make convolution_bin CONVOLUTION=0 MODE=0 TARGET=$@ CFLAGSLIBS="`pkg-config --libs --cflags fftw3`"
convolution_fftw_benchmark_circular.bin: convolution_benchmark.cc convolution_fftw.h
	make convolution_bin CONVOLUTION=0 MODE=1 TARGET=$@ CFLAGSLIBS="`pkg-config --libs --cflags fftw3`"
convolution_fftw_benchmark_linear_unpadded.bin: convolution_benchmark.cc convolution_fftw.h
	make convolution_bin CONVOLUTION=0 MODE=2 TARGET=$@ CFLAGSLIBS="`pkg-config --libs --cflags fftw3`" 
convolution_fftw_benchmark_circular_unpadded.bin: convolution_benchmark.cc convolution_fftw.h
	make convolution_bin CONVOLUTION=0 MODE=3 TARGET=$@ CFLAGSLIBS="`pkg-config --libs --cflags fftw3`"


convolution_gsl_benchmark_linear.bin: convolution_benchmark.cc convolution_gsl.h
	make convolution_bin CONVOLUTION=1 MODE=0 TARGET=$@ CFLAGSLIBS="`pkg-config --libs --cflags gsl`"
convolution_gsl_benchmark_circular.bin: convolution_benchmark.cc convolution_gsl.h
	make convolution_bin CONVOLUTION=1 MODE=1 TARGET=$@ CFLAGSLIBS="`pkg-config --libs --cflags gsl`"
convolution_gsl_benchmark_linear_unpadded.bin: convolution_benchmark.cc convolution_gsl.h
	make convolution_bin CONVOLUTION=1 MODE=2 TARGET=$@ CFLAGSLIBS="`pkg-config --libs --cflags gsl`" 
convolution_gsl_benchmark_circular_unpadded.bin: convolution_benchmark.cc convolution_gsl.h
	make convolution_bin CONVOLUTION=1 MODE=3 TARGET=$@ CFLAGSLIBS="`pkg-config --libs --cflags gsl` "


convolution_std_benchmark_linear.bin: convolution_benchmark.cc convolution_std.h
	make convolution_bin CONVOLUTION=2 MODE=0 TARGET=$@ CFLAGSLIBS=""
convolution_std_benchmark_circular.bin: convolution_benchmark.cc convolution_std.h
	make convolution_bin CONVOLUTION=2 MODE=1 TARGET=$@ CFLAGSLIBS=""

convolution_octave_benchmark_linear.bin: convolution_benchmark.cc
	make convolution_bin CONVOLUTION=3 MODE=0 TARGET=$@ CFLAGSLIBS="-std=c++11 `mkoctfile -p ALL_CXXFLAGS` `mkoctfile -p OCTAVE_LIBS` `mkoctfile -p LFLAGS`"
convolution_octave_benchmark_circular.bin: convolution_benchmark.cc
	make convolution_bin CONVOLUTION=3 MODE=1 TARGET=$@ CFLAGSLIBS="-std=c++11 `mkoctfile -p ALL_CXXFLAGS` `mkoctfile -p OCTAVE_LIBS` `mkoctfile -p LFLAGS`"

convolution_test_bin:
	g++ -o $(TARGET) convolution_test.cc -DCONVOLUTION=$(CONVOLUTION) -O3 -Wall $(CFLAGSLIBS) -std=c++11 `mkoctfile -p ALL_CXXFLAGS` `mkoctfile -p OCTAVE_LIBS` `mkoctfile -p LFLAGS`

convolution_test_fftw.bin:convolution_test.cc convolution_fftw.h
	make convolution_test_bin TARGET=$@ CONVOLUTION=0 CFLAGSLIBS="`pkg-config --libs --cflags fftw3`"

convolution_test_gsl.bin:convolution_test.cc convolution_gsl.h
	make convolution_test_bin TARGET=$@ CONVOLUTION=1 CFLAGSLIBS="`pkg-config --libs --cflags gsl`"

convolution_test_std.bin:convolution_test.cc convolution_std.h
	make convolution_test_bin TARGET=$@ CONVOLUTION=2 CFLAGSLIBS=""


clean : 
	rm -f *.o *~ *.bak *.log *.m
	rm -f $(BINS)
