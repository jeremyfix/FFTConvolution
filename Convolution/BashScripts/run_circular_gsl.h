#!/bin/bash

rm ../Data/benchmarks_CircularConvolution_gsl.txt

cd ../bin
for i in $(seq 3 512)
do
    for j in $(seq 3 $(($i-1)))
    do
	./circular_convolution_gsl_benchmark.gsl.bin $i $j
    done
done


