#!/bin/bash

rm ../Data/benchmarks_convolution_gsl.txt

MAX_SIZE=100

cd ../bin
echo "Running convolution_gsl_benchmark_linear.bin"
for i in $(seq 3 $MAX_SIZE)
do
    for j in $(seq 3 $(($i-1)))
    do
	./convolution_gsl_benchmark_linear.bin $i $j > /dev/null
    done
done
echo "Running convolution_gsl_benchmark_linear_optimal.bin"
for i in $(seq 3 $MAX_SIZE)
do
    for j in $(seq 3 $(($i-1)))
    do
	./convolution_gsl_benchmark_linear_optimal.bin $i $j > /dev/null
    done
done


echo "Running convolution_gsl_benchmark_circular.bin"
for i in $(seq 3 $MAX_SIZE)
do
    for j in $(seq 3 $(($i-1)))
    do
	./convolution_gsl_benchmark_circular.bin $i $j > /dev/null
    done
done
echo "Running convolution_gsl_benchmark_circular_optimal.bin"
for i in $(seq 3 $MAX_SIZE)
do
    for j in $(seq 3 $(($i-1)))
    do
	./convolution_gsl_benchmark_circular_optimal.bin $i $j > /dev/null
    done
done

cd ../BashScripts


