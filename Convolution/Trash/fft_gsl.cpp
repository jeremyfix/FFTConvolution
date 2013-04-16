// // Compute a 2D FFT with GSL
// Compile with g++ -o fft_gsl fft_gsl.cpp -O3 `pkg-config --libs --cflags gsl`

#include <iostream>
#include <cstdio>
#include <cmath>
#include <cassert>

// To Load an image
#include "CImg.h"

// To use the GSL FFT
#include <gsl/gsl_blas.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_fft_complex.h>

// To measure execution time
#include <sys/time.h>

#define NB_REPETITIONS 500

/*****************************************************************/

void compute_fft_gsl(gsl_matrix * src, gsl_matrix * dst)
{
    int height = src->size1;
    int width = src->size2;

    // Create some required matrices
    gsl_fft_complex_workspace * ws_column = gsl_fft_complex_workspace_alloc(height);
    gsl_fft_complex_workspace * ws_line = gsl_fft_complex_workspace_alloc(width);
    gsl_fft_complex_wavetable * wv_column = gsl_fft_complex_wavetable_alloc(height);
    gsl_fft_complex_wavetable * wv_line = gsl_fft_complex_wavetable_alloc(width);

    // Copy the real part and let the imaginary part to 0
    // The GSL works in place, directly on the dst matrix
    gsl_matrix_set_zero(dst);
    gsl_vector_memcpy(&gsl_vector_view_array_with_stride(dst->data,2,height * width).vector, &gsl_vector_view_array(src->data,width*height).vector);

    int i, j;
    for(i = 0 ; i < height ; ++i)
    {
        // Apply the FFT on the line i
        gsl_fft_complex_forward (&dst->data[i*2*width],1 , width, wv_line, ws_line);
    }

    for(j = 0 ; j < width ; ++j)
    {
        // Apply the FFT on the column j
        gsl_fft_complex_forward (&dst->data[2*j],width, height, wv_column, ws_column);
    }

    gsl_fft_complex_workspace_free(ws_column);
    gsl_fft_complex_workspace_free(ws_line);
    gsl_fft_complex_wavetable_free(wv_column);
    gsl_fft_complex_wavetable_free(wv_line);
}

int main(int argc, char * argv[])
{
    int i,j;

    if(argc != 2)
      {
	printf("Usage : fft_gsl <N>   for computing the FFT of an image of size NxN\n");
	return -1;
      }

    int w_src = atoi(argv[1]);
    int h_src = atoi(argv[1]);
    gsl_matrix * src = gsl_matrix_alloc(h_src, w_src);
    for(i = 0 ; i < h_src ; ++i)
        for(j = 0 ; j < w_src ; ++j)
            gsl_matrix_set(src,i ,j, rand()/double(RAND_MAX));

    // Variables for measuring the execution time
    struct timeval before,after;
    double sbefore, safter, total;

    printf("Image size : %i x %i \n", h_src , w_src);
    
    // Create some matrices holding the image and the resulting FFT
    // The FFTs are complex, we therefore need 2 times the number of pixels in the image
    gsl_matrix * dst_fft_gsl = gsl_matrix_alloc(h_src,2*w_src);

    // Perform the 2D FFT with the GSL
    gettimeofday(&before, NULL);
    for(i = 0 ; i < NB_REPETITIONS ; ++i)
        compute_fft_gsl(src, dst_fft_gsl);
    gettimeofday(&after, NULL);
    sbefore = before.tv_sec + before.tv_usec * 1E-6;
    safter =after.tv_sec + after.tv_usec * 1E-6;
    total = safter - sbefore;
    printf("Execution time with GSL : %e s.\n", total/NB_REPETITIONS);

    // Free the memory
    gsl_matrix_free(src);
    gsl_matrix_free(dst_fft_gsl);
}
