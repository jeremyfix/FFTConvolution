#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>

//#include <complex.h>
//#include <fftw3.h>

#include <string>
#include <fstream>

#define VERBOSE true
#define SAVE_RESULTS false

// To measure the execution time
#include <sys/time.h>

// The code for the convolutions
#include "convolution_fftw.h"

#define NB_REPETITIONS 20

/* ******************************************************************************************************** */
/*                   Main : it loads an image and convolves it with a gaussian filter                       */
/* ******************************************************************************************************** */

int main(int argc, char * argv[])
{
    if(argc != 3)
    {
        printf("Usage : %s <img_size> <kernel_size>\n", argv[0]);
        printf(" It performs a linear convolution using : \n");
        printf(" 1 - The FFTW without padding \n");
        printf(" 2 - The FFTW with padding \n");
        return -1;
    }

    std::ofstream results;
    if(SAVE_RESULTS) results.open("../Data/benchmarks_LinearConvolution_fftw.txt", std::ios::app);

    struct timeval before,after;
    double sbefore, safter, total;
    int h_src, w_src;
    int h_kernel, w_kernel;
    int small_h_kernel = 3;
    int small_w_kernel = 3;

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

    // And compute the linear convolution
    if(VERBOSE) printf("Execution times : \n");

    // Initialize the workspace for performing the convolution
    // This workspace can be kept until the size of the
    // image changes
    FFTW_Workspace ws;
    init_workspace_fftw(ws,         FFTW_LINEAR        , h_src,w_src,h_kernel,w_kernel);
    FFTW_Workspace ws_optimal;
    init_workspace_fftw(ws_optimal, FFTW_LINEAR_OPTIMAL, h_src,w_src,h_kernel,w_kernel);

    gettimeofday(&before, NULL);
    for(int i = 0 ; i < NB_REPETITIONS ; ++i)
      linear_convolution_fft_fftw(ws, src, h_src, w_src, kernel, h_kernel, w_kernel, dst);
    gettimeofday(&after, NULL);
    sbefore = before.tv_sec + before.tv_usec * 1E-6;
    safter =after.tv_sec + after.tv_usec * 1E-6;
    total = safter - sbefore;
    if(VERBOSE) printf("Unpadded FFTW : %e s.\n", total/NB_REPETITIONS);
    if(SAVE_RESULTS) results << total/NB_REPETITIONS << " ";

    // And compute the linear convolution with an optimal size
    gettimeofday(&before, NULL);
    for(int i = 0 ; i < NB_REPETITIONS ; ++i)
        linear_convolution_fft_fftw_optimal(ws_optimal, src, h_src, w_src, kernel, h_kernel, w_kernel, dst_optimal);
    gettimeofday(&after, NULL);
    sbefore = before.tv_sec + before.tv_usec * 1E-6;
    safter =after.tv_sec + after.tv_usec * 1E-6;
    total = safter - sbefore;
    if(VERBOSE) printf("Padded FFTW : %e \n", total/NB_REPETITIONS);
    if(SAVE_RESULTS) results << total/NB_REPETITIONS << std::endl;

    delete[] src;
    delete[] dst;
    delete[] dst_optimal;
    delete[] kernel;

    clear_workspace_fftw(ws);
    clear_workspace_fftw(ws_optimal);
}

