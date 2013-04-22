#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <string>
#include <fstream>

#define VERBOSE true
#define SAVE_RESULTS false

// To measure the execution time
#include <sys/time.h>

// The code for the convolution
#include "convolution_fftw.h"
using namespace FFTW_Convolution;

#define NB_REPETITIONS 20

/******************************************************/
/*                      Main                          */
/******************************************************/

int main(int argc, char * argv[])
{
    if(argc != 3)
    {
        printf("Usage : circular_convolution_fftw_benchmark <img_size> <kernel_size>\n");
        printf(" Performs a 2D circular convolution using FFTW3 and                 \n");
	printf("  1 - The FFTW without padding                                      \n");
	printf("  2 - The FFTW with padding                                         \n");
        return -1;
    }

    std::ofstream results;
    if(SAVE_RESULTS) results.open("../Data/benchmarks_CircularConvolution_fftw.txt", std::ios::app);

    struct timeval before,after;
    double sbefore, safter, total;
    int h_src, w_src;
    int h_kernel, w_kernel;

    h_src = w_src = atoi(argv[1]);
    h_kernel = w_kernel = atoi(argv[2]);

    if(VERBOSE) printf("Image size : %i %i \n", h_src, w_src);
    if(VERBOSE) printf("Kernel size : %i %i \n", h_kernel, w_kernel);
    if(SAVE_RESULTS) results << std::scientific << h_src << " " << h_kernel << " ";

    double * src = new double[h_src*w_src];
    for(int i = 0 ; i < h_src ; ++i)
        for(int j = 0 ; j < w_src ; ++j)
            src[i*w_src+j]=rand()/double(RAND_MAX);

    double * kernel = new double[h_kernel*w_kernel];
    for(int i = 0 ; i < h_kernel ; ++i)
        for(int j = 0 ; j < w_kernel ; ++j)
            kernel[i*w_kernel+j] = rand()/double(RAND_MAX);

    double * dst = new double[h_src*w_src];
    double * dst_optimal = new double[h_src*w_src];

    // And compute the circular convolution
    if(VERBOSE) printf("Execution times : \n");

    // Initialize the workspace for performing the convolution
    // This workspace can be kept until the size of the
    // image changes
    Workspace ws;
    init_workspace(ws,         CIRCULAR,         h_src, w_src, h_kernel, w_kernel);

    gettimeofday(&before, NULL);
    for(int i = 0 ; i < NB_REPETITIONS ; ++i)
        convolve(ws, src, kernel, dst);
    gettimeofday(&after, NULL);
    sbefore = before.tv_sec + before.tv_usec * 1E-6;
    safter =after.tv_sec + after.tv_usec * 1E-6;
    total = safter - sbefore;
    if(VERBOSE) printf("Unpadded FFTW : %e s. \n", total/NB_REPETITIONS);
    if(SAVE_RESULTS) results << total/NB_REPETITIONS << " ";

    // Update the convolution mode
    update_workspace(ws, CIRCULAR_OPTIMAL, h_src, w_src, h_kernel, w_kernel);

    // And compute the circular convolution with an optimal size
    gettimeofday(&before, NULL);
    for(int i = 0 ; i < NB_REPETITIONS ; ++i)
        convolve(ws, src, kernel, dst_optimal);
    gettimeofday(&after, NULL);
    sbefore = before.tv_sec + before.tv_usec * 1E-6;
    safter =after.tv_sec + after.tv_usec * 1E-6;
    total = safter - sbefore;
    if(VERBOSE) printf("Padded fftw : %e s. \n", total/NB_REPETITIONS);
    if(SAVE_RESULTS) results << total/NB_REPETITIONS << " ";

    if(SAVE_RESULTS) results << std::endl;

    delete[] src;
    delete[] kernel;
    delete[] dst;
    delete[] dst_optimal;

    clear_workspace(ws);
}
