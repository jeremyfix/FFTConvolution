#/bin/bash
echo "Running linear convolution GSL"
./run_linear_gsl.sh
echo "Running linear convolution FFTW"
./run_linear_fftw.sh
echo "Running circular convolution GSL"
./run_circular_gsl.sh
echo "Running circular convolution FFTW"
./run_circular_fftw.sh
echo "Running standard convolution"
./run_std_bench.sh

