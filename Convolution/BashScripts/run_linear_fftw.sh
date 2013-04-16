#!/bin/bash

rm ../Data/benchmarks_LinearConvolution_fftw.txt

cd ../bin
for i in $(seq 3 512)
do
    for j in $(seq 3 $(($i-1)))
    do
	./linear_convolution_fftw_benchmark.fftw.bin $i $j
    done
done


