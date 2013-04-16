#!/bin/bash

rm benchmarks_CircularConvolution_fftw.txt


for i in $(seq 3 512)
do
    unbuffer ./circular_convolution_fftw_benchmark $i $i
done


