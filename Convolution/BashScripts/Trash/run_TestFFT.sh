rm benchmarks_TestFFT.txt

touch benchmarks_TestFFT.txt
for i in $(seq 10 1 400)
do
    unbuffer ./test_fft $i >> benchmarks_TestFFT.txt
done
