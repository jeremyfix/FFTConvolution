all: help

help:
	@echo "Targets :"
	@echo " make test : run the comparisons of our libs against octave and matlab"
	@echo " make benchmarks : executes the benchmarks of the different convolution implementations"
	@echo " make plots : plot and fits the execution times obtained with the benchmarks"
	@echo " make clean : clear the directory "
	@echo " make doc  : compile the Tex document"

test:
	cd src && make test

benchmarks: 
	cd src && make benchmarks

plots:
	cd PythonScripts && make plots

clean:
	cd Doc && make clean
	cd src && make clean
	cd Data && make clean

doc:
	cd Doc && make


