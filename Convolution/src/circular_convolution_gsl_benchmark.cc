#include <iostream>
#include <cstdio>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_fft_complex.h>
#include <gsl/gsl_fft_real.h>
#include <gsl/gsl_fft_halfcomplex.h>
#include <cmath>
#include <cassert>

#include <string>
#include <fstream>

// The code for the convolution
#include "convolution_gsl.h"

#define VERBOSE false
#define SAVE_RESULTS true

// To measure the execution time
#include <sys/time.h>

#define NB_REPETITIONS 20

/******************************************************/
/*                      Main                          */
/******************************************************/

int main(int argc, char * argv[])
{
    if(argc != 3)
    {
        printf("Usage : circular_convolution_gsl_benchmark <img_size> <kernel_size>\n");
        printf(" Performs a 2D circular convolution using GSL \n");
        return -1;
    }

    std::ofstream results;
    if(SAVE_RESULTS) results.open("../Data/benchmarks_CircularConvolution_gsl.txt", std::ios::app);

    struct timeval before,after;
    double sbefore, safter, total;
    int h_src, w_src;
    int h_kernel, w_kernel;

    h_src = w_src = atoi(argv[1]);
    h_kernel = w_kernel = atoi(argv[2]);

    if(VERBOSE) printf("Image size : %i %i \n", h_src, w_src);
    if(VERBOSE) printf("Kernel size : %i %i \n", h_kernel, w_kernel);
    if(SAVE_RESULTS) results << std::scientific << h_src << " " << h_kernel << " ";

    gsl_matrix * src = gsl_matrix_alloc(h_src, w_src);
    for(int i = 0 ; i < h_src ; ++i)
        for(int j = 0 ; j < w_src ; ++j)
            gsl_matrix_set(src,i ,j, rand()/double(RAND_MAX));

    gsl_matrix * kernel = gsl_matrix_alloc(h_kernel, w_kernel);
    for(int i = 0 ; i < h_kernel ; ++i)
        for(int j = 0 ; j < w_kernel ; ++j)
            gsl_matrix_set(kernel,i ,j, rand()/double(RAND_MAX));

    gsl_matrix * dst = gsl_matrix_alloc(h_src, w_src);
    gsl_matrix * dst_optimal = gsl_matrix_alloc(h_src, w_src);
    gsl_matrix * dst_combined = gsl_matrix_alloc(h_src, w_src);

    // And compute the circular convolution
    if(VERBOSE) printf("Execution times : \n");

    gettimeofday(&before, NULL);
    /*for(int i = 0 ; i < NB_REPETITIONS ; ++i)
        circular_convolution_fft_gsl(src, kernel, dst);
    */
    gettimeofday(&after, NULL);
    sbefore = before.tv_sec + before.tv_usec * 1E-6;
    safter =after.tv_sec + after.tv_usec * 1E-6;
    total = safter - sbefore;
    if(VERBOSE) printf("Unpadded GSL : %e s. \n", total/NB_REPETITIONS);
    if(SAVE_RESULTS) results << total/NB_REPETITIONS << " ";

    // And compute the circular convolution with an optimal size
    gettimeofday(&before, NULL);
    for(int i = 0 ; i < NB_REPETITIONS ; ++i)
        circular_convolution_fft_gsl_optimal(src, kernel, dst_optimal);
    gettimeofday(&after, NULL);
    sbefore = before.tv_sec + before.tv_usec * 1E-6;
    safter =after.tv_sec + after.tv_usec * 1E-6;
    total = safter - sbefore;
    if(VERBOSE) printf("Padded GSL : %e s. \n", total/NB_REPETITIONS);
    if(SAVE_RESULTS) results << total/NB_REPETITIONS << " ";

    // And compute the circular convolution combining the two previous
    // computations with the following rule :
    // Don't use padding when the size is optimal
    // Use padding otherwise
    gettimeofday(&before, NULL);
    for(int i = 0 ; i < NB_REPETITIONS ; ++i)
        circular_convolution_fft_gsl_combined(src, kernel, dst_combined);
    gettimeofday(&after, NULL);
    sbefore = before.tv_sec + before.tv_usec * 1E-6;
    safter =after.tv_sec + after.tv_usec * 1E-6;
    total = safter - sbefore;
    if(VERBOSE) printf("Combined GSL : %e s. \n", total/NB_REPETITIONS);
    if(SAVE_RESULTS) results << total/NB_REPETITIONS << " ";


    if(SAVE_RESULTS) results << std::endl;

    // Free the memory
    gsl_matrix_free(src);
    gsl_matrix_free(kernel);
    gsl_matrix_free(dst);
    gsl_matrix_free(dst_optimal);
    gsl_matrix_free(dst_combined);
}
