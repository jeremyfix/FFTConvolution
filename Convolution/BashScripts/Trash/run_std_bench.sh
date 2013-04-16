#!/bin/bash

rm benchmarks_StdConvolution.txt

for i in {64..512}
do
    for j in $(seq 3 31)
    do
	echo "$i $j " $(unbuffer ./std_convolution $i $j) >> benchmarks_StdConvolution.txt
    done
done


