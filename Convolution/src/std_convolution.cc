#include <iostream>
#include <cstdio>
#include <cstdlib>
//#include <cmath>
//#include <ctgmath>
#include <cstring>
#include <fstream>

#define VERBOSE true
#define SAVE_RESULTS false

// To measure the execution time
#include <sys/time.h>

// The code for the convolution
#include "convolution_std.h"

#define NB_REPETITIONS 20

/* ******************************************************************************************************** */
/*                   Main : it creates a two random 2D signals and compute their convolution                */
/*                          using nested for loops                                                          */
/* ******************************************************************************************************** */

int main(int argc, char * argv[])
{
    if(argc != 3)
    {
        printf("Usage : std_convolution <img_size> <kernel_size>\n");
        printf(" It performs a convolution of an image of size img_size x img_size \n");
        printf(" by a kernel of size kernel_size x kernel_size using nested for loops\n");
        return -1;
    }

    std::ofstream results;
    char * filename = "../Data/benchmarks_std_convolution.txt";
    if(SAVE_RESULTS) results.open(filename, std::ios::app);

    struct timeval before,after;
    double sbefore, safter, total;

    int h_src, w_src, h_kernel, w_kernel;

    h_src = atoi(argv[1]);
    w_src = h_src;
    h_kernel = atoi(argv[2]);
    w_kernel = h_kernel;
    if(VERBOSE) printf("Image of size %i x %i , kernel of size %i x %i \n", h_src, w_src, h_kernel,w_kernel);
    if(SAVE_RESULTS) printf(" The results are saved in %s \n", filename);
    if(SAVE_RESULTS) results << h_src << " " << h_kernel << " " << std::scientific;

    // Build random images to convolve
    double * src = new double[h_src*w_src];
    for(int i = 0 ; i < h_src ; ++i)
        for(int j = 0 ; j < w_src ; ++j)
            src[i*w_src+j]=rand()/double(RAND_MAX);

    double * kernel = new double[h_kernel*w_kernel];
    for(int i = 0 ; i < h_kernel ; ++i)
        for(int j = 0 ; j < w_kernel ; ++j)
            kernel[i*w_kernel+j] = rand()/double(RAND_MAX);

    double * dst = new double[h_src*w_src];
    double * dst_circ = new double[h_src*w_src];

    // And compute the linear convolution
    gettimeofday(&before, NULL);
    for(int i = 0 ; i < NB_REPETITIONS ; ++i)
        std_linear_convolution(src, h_src, w_src, kernel, h_kernel, w_kernel, dst);
    gettimeofday(&after, NULL);
    sbefore = before.tv_sec + before.tv_usec * 1E-6;
    safter =after.tv_sec + after.tv_usec * 1E-6;
    total = safter - sbefore;
    if(VERBOSE) printf("Execution time for a linear convolution : %e s.\n", total/NB_REPETITIONS);
    if(SAVE_RESULTS) results << total/NB_REPETITIONS << " ";

    // And compute the circular convolution
    gettimeofday(&before, NULL);
    for(int i = 0 ; i < NB_REPETITIONS ; ++i)
        std_circular_convolution(src, h_src, w_src, kernel, h_kernel, w_kernel, dst_circ);
    gettimeofday(&after, NULL);
    sbefore = before.tv_sec + before.tv_usec * 1E-6;
    safter =after.tv_sec + after.tv_usec * 1E-6;
    total = safter - sbefore;
    if(VERBOSE) printf("Execution time for a circular convolution : %e s.\n", total/NB_REPETITIONS);
    if(SAVE_RESULTS) results << total/NB_REPETITIONS << std::endl;

    delete[] src;
    delete[] dst;
    delete[] dst_circ;
    delete[] kernel;
}

