#!/bin/bash
echo "linear GSL benchmark.."
unbuffer ./linear_convolution_gsl_benchmark > benchmarks_LinearConvolution_gsl.txt
echo "linear FFTW benchmark.."
unbuffer ./linear_convolution_fftw_benchmark > benchmarks_LinearConvolution_fftw.txt
echo "Circular GSL benchmark.."
unbuffer ./circular_convolution_gsl_benchmark > benchmarks_CircularConvolution_gsl.txt
echo "Circular FFTW benchmark.."
unbuffer ./circular_convolution_fftw_benchmark > benchmarks_CircularConvolution_fftw.txt
