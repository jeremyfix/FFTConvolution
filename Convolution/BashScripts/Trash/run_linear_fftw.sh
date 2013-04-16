#!/bin/bash

rm benchmarks_LinearConvolution_fftw.txt


for i in $(seq 3 512)
do
    unbuffer ./linear_convolution_fftw_benchmark $i $i
done


