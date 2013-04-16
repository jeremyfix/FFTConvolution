#!/bin/bash

rm ../Data/benchmarks_std_convolution.txt

cd ../bin
for i in $(seq 3 512)
do
    for j in $(seq 3 $(($i-1)))
    do
	./std_convolution.bin $i $j
    done
done


