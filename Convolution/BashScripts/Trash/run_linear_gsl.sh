#!/bin/bash

rm benchmarks_LinearConvolution_gsl.txt


for i in $(seq 3 512)
do
    unbuffer ./linear_convolution_gsl_benchmark $i $i
done


