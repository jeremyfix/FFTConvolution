#!/bin/bash

rm benchmarks_CircularConvolution_gsl.txt


for i in $(seq 3 512)
do
    unbuffer ./circular_convolution_gsl_benchmark $i $i
done


