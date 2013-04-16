#!/bin/bash

rm benchmarks_linear_everything.txt


# Choosen sizes : 128 256  512 1024
# Filter sizes : 3x3 7x7 11x11 15x15 19x19 23x23 27x27
for i in $(seq 7 10)
do
    img_size=$(echo "2^$i" | bc) 
    for j in $(seq 3 4 27)
    do
	unbuffer ./linear_everything $img_size $j >> benchmarks_linear_everything.txt
    done
done


