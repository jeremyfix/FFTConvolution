#!/bin/bash

rm benchmarks_LinearConvolution_fftw_all.txt


for i in $(seq 3 200)
do
    upperj=$(($i-1))
    for j in $(seq 3 $upperj)
    do
	./linear_convolution_fftw_benchmark $i $j
    done
done


