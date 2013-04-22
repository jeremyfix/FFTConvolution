// Compilation with :
// g++ -o convolution_benchmark convolution_benchmark.cc -DVERBOSE=true -DSAVE_RESULTS=false -DCONVOLUTION=0 -DMODE=0 -O3 -Wall `pkg-config --libs --cflags gsl fftw3`
// Modes :
// - 0 : LINEAR, 
// - 1 : LINEAR_OPTIMAL, 
// - 2 : CIRCULAR, 
// - 3 : CIRCULAR_OPTIMAL
// Convolution:
// - 0 : fftw, 
// - 1 : gsl, 
// - 2 : std

// Using -DCONVOLUTION=2 , only the linear and circular modes are available, compilation should complain otherwise

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <string>
#include <fstream>
// To measure the execution time
#include <sys/time.h>

#ifndef VERBOSE
#define VERBOSE true
#endif

#ifndef SAVE_RESULTS
#define SAVE_RESULTS false
#endif

#if CONVOLUTION==0
#include "convolution_fftw.h"
using namespace FFTW_Convolution;
#define filename_results "../Data/benchmarks_convolution_fftw.txt"

#elif CONVOLUTION==1
#include "convolution_gsl.h"
using namespace GSL_Convolution;
#define filename_results "../Data/benchmarks_convolution_gsl.txt"

#elif CONVOLUTION==2
#include "convolution_std.h"
using namespace STD_Convolution;
#define filename_results "../Data/benchmarks_convolution_std.txt"

#else
# error Unrecognized convolution type!
#endif

#if MODE==0
#define MODE_STR "linear"
#define CONVOLUTION_MODE LINEAR
#elif MODE==1
#define MODE_STR "linear_optimal"
#define CONVOLUTION_MODE LINEAR_OPTIMAL
#elif MODE==2
#define MODE_STR "circular"
#define CONVOLUTION_MODE CIRCULAR
#elif MODE==3
#define MODE_STR "circular_optimal"
#define CONVOLUTION_MODE CIRCULAR_OPTIMAL
#endif


#define NB_REPETITIONS 20

/* ******************************************************************************************************** */
/*                   Main : it loads an image and convolves it with a gaussian filter                       */
/* ******************************************************************************************************** */

int main(int argc, char * argv[])
{
    if(argc != 3)
    {
        printf("Usage : %s <img_size> <kernel_size>\n", argv[0]);
	printf("It measures the execution time for convolving \n");
	printf("a source of size img_size x img_size with a kernel of size kernel_size x kernel_size\n");
        return -1;
    }

    std::ofstream results;
    if(SAVE_RESULTS) results.open(filename_results, std::ios::app);

    struct timeval before,after;
    double sbefore, safter, total;
    int h_src, w_src;
    int h_kernel, w_kernel;

    h_src = w_src = atoi(argv[1]);
    h_kernel = w_kernel = atoi(argv[2]);

    if(VERBOSE) printf("Image size : %i %i \n", h_src, w_src);
    if(VERBOSE) printf("Kernel size : %i %i \n", h_kernel, w_kernel);
    if(SAVE_RESULTS) results << MODE_STR << " " << std::scientific << h_src << " " << h_kernel << " ";

    // Initialization of the source and kernel
    double * src = new double[h_src*w_src];
    for(int i = 0 ; i < h_src ; ++i)
        for(int j = 0 ; j < w_src ; ++j)
            src[i*w_src+j]=rand()/double(RAND_MAX);

    double * kernel = new double[h_kernel*w_kernel];
    for(int i = 0 ; i < h_kernel ; ++i)
        for(int j = 0 ; j < w_kernel ; ++j)
            kernel[i*w_kernel+j] = rand()/double(RAND_MAX);

    double * dst = new double[h_src*w_src];

    // And compute the linear convolution
    if(VERBOSE) printf("Execution times : \n");

    // Initialize the workspace for performing the convolution
    // This workspace can be kept until the size of the
    // image changes
    Workspace ws;
    init_workspace(ws, CONVOLUTION_MODE, h_src, w_src, h_kernel, w_kernel);

    gettimeofday(&before, NULL);
    // The main loop
    for(int i = 0 ; i < NB_REPETITIONS ; ++i)
      convolve(ws, src, kernel, dst);
    //
    gettimeofday(&after, NULL);
    sbefore = before.tv_sec + before.tv_usec * 1E-6;
    safter =after.tv_sec + after.tv_usec * 1E-6;
    total = safter - sbefore;
    if(VERBOSE) printf("%e s.\n", total/NB_REPETITIONS);
    if(SAVE_RESULTS) results << total/NB_REPETITIONS << "\n";
    if(SAVE_RESULTS) results.close();

    // Clean up
    delete[] src;
    delete[] dst;
    delete[] kernel;

    clear_workspace(ws);
}
