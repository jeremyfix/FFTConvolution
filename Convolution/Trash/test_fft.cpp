// In this file we propose 2 implementations of a 2D FFT using the GSL and FFTW3
// Compile with g++ -o test_fft test_fft.cpp -O3 `pkg-config --libs --cflags gsl fftw3`

#include <iostream>
#include <cstring>
#include <cmath>

// To use the GSL FFT
#include <gsl/gsl_blas.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_fft_complex.h>

// To use FFTW3
#include <fftw3.h>

// To measure execution time
#include <sys/time.h>

#define NB_REPETITIONS 50

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

void compute_fft_fftw3(double * src, int height, int width, double * dst)
{
    // Create the required objects for FFTW
    fftw_complex *in_src, *out_src;
    fftw_plan p_forw;

    in_src = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * height * width);
    out_src = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * height * width);
    p_forw = fftw_plan_dft_2d(height, width, in_src, out_src, FFTW_FORWARD, FFTW_ESTIMATE);

    for(int i = 0 ; i < height ; ++i)
    {
        for(int j = 0 ; j < width ; ++j)
        {
            in_src[i * width + j][0] = src[i*width+j];
            in_src[i * width + j][1] = 0.0;
        }
    }

    // Compute the forward fft
    fftw_execute(p_forw);

    // Now we just need to copy the right part of out_src into dst
    for(int i  = 0 ; i < height ; ++i)
    {
        for(int j = 0 ; j < width ; ++j)
        {
            dst[i*2*width + 2*j] = out_src[i * width + j][0];
            dst[i*2*width + 2*j+1] = out_src[i * width + j][1];
        }
    }

    fftw_destroy_plan(p_forw);
    fftw_free(in_src);
    fftw_free(out_src);
}

int main(int argc, char * argv[])
{

    if(argc != 2)
    {
        printf("Usage : test_fft <signal_size> \n");
        printf(" It computes a 2D FFT of an image of size signal_size x signal_size \n");
    }

    int i,j;

    int w_src = atoi(argv[1]);
    int h_src = atoi(argv[1]);

    // Src is the source matrix when using the gsl
    gsl_matrix * src = gsl_matrix_alloc(h_src, w_src);
    // src_array is the source double array when using FFTW3
    double * src_array = new double[h_src * w_src];

    for(i = 0 ; i < h_src ; ++i)
        for(j = 0 ; j < w_src ; ++j)
            gsl_matrix_set(src,i ,j, rand()/double(RAND_MAX));
    // Copy the generated data in src_array
    memcpy(src_array, src->data, sizeof(double) * h_src * w_src);

    // Variables for measuring the execution time
    struct timeval before,after;
    double sbefore, safter, total;

    // Create some matrices holding the image and the resulting FFT
    // The FFTs are complex, we therefore need 2 times the number of pixels in the image
    gsl_matrix * dst_fft_gsl = gsl_matrix_alloc(h_src,2*w_src);
    double * dst_fft_fftw3 = new double[h_src * 2 * w_src];

    // Print the image size
    printf("Signal size : %i x %i \n", h_src, w_src);

    // Perform the 2D FFT with the GSL
    gettimeofday(&before, NULL);
    for(i = 0 ; i < NB_REPETITIONS ; ++i)
        compute_fft_gsl(src, dst_fft_gsl);
    gettimeofday(&after, NULL);
    sbefore = before.tv_sec + before.tv_usec * 1E-6;
    safter =after.tv_sec + after.tv_usec * 1E-6;
    total = safter - sbefore;
    // Print the execution time
    printf("Execution time with the GSL : %e \n", total/NB_REPETITIONS);

    // Perform the 2D FFT with FFTW3
    gettimeofday(&before, NULL);
    for(i = 0 ; i < NB_REPETITIONS ; ++i)
        compute_fft_fftw3(src_array, h_src, w_src, dst_fft_fftw3);
    gettimeofday(&after, NULL);
    sbefore = before.tv_sec + before.tv_usec * 1E-6;
    safter =after.tv_sec + after.tv_usec * 1E-6;
    total = safter - sbefore;
    printf("Execution time with FFTW3 : %e s.\n", total/NB_REPETITIONS);

    // Compare the outputs
    double error_value;
    gsl_vector * diff = gsl_vector_alloc(h_src * 2 * w_src);

    gsl_vector_memcpy(diff, &gsl_vector_view_array(dst_fft_gsl->data, h_src * 2 * w_src).vector);
    gsl_vector_sub(diff, &gsl_vector_view_array(dst_fft_fftw3, h_src * 2*w_src).vector);
    error_value = sqrt(gsl_pow_2(gsl_blas_dnrm2(diff))/(double(diff->size)));
    printf("RMS error : %e \n", error_value);

    // Print the line with the size, measured times and RMS

    // Free the memory
    gsl_matrix_free(src);
    gsl_matrix_free(dst_fft_gsl);
    gsl_vector_free(diff);

    delete[] src_array;
    delete[] dst_fft_fftw3;
}
