#include <iostream>
#include <cstdlib>
#include <cmath>

// To use FFTW3
#include <fftw3.h>

// To measure execution time
#include <sys/time.h>

#define NB_REPETITIONS 50

/*****************************************************************/
// Computing a single DFT 

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

/*****************************************************************/
// Computing 2 DFT at once

void compute_2fft_fftw3(double * src1, double * src2, int height, int width, double * dst1, double * dst2)
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
            in_src[i * width + j][0] = src1[i*width+j];
            in_src[i * width + j][1] = src2[i*width+j];
        }
    }

    // Compute the forward fft
    fftw_execute(p_forw);

    // Now we just need to copy the right part of out_src into dst
    double re_h, im_h, re_hs, im_hs;

    for(int i  = 0 ; i < height ; ++i)
    {
        for(int j = 0 ; j < width ; ++j)
        {
            re_h = out_src[i * width + j][0];
            im_h = out_src[i * width + j][1];
            re_hs = out_src[ ((width-i) % width) * width + ((height-j)%height)][0];
            im_hs = out_src[ ((width-i) % width) * width + ((height-j)%height)][1];

            dst1[i*2*width + 2*j] = 0.5 * (re_h + re_hs);
            dst1[i*2*width + 2*j+1] = 0.5 * (im_h - im_hs);
            dst2[i*2*width + 2*j] = 0.5 * (im_h + im_hs);
            dst2[i*2*width + 2*j+1] = 0.5*(- re_h + re_hs);
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
        printf("Usage : 2DFT <img_size>\n");
        printf(" This script compares the computation of 2 DFT separately and 2 DFT at once \n");
        return -1;
    }

    int i,j;

    int w_src = atoi(argv[1]);
    int h_src = atoi(argv[1]);
    double *src1 = new double[h_src*w_src];
    double *src2 = new double[h_src*w_src];
    
    for(i = 0 ; i < h_src ; ++i)
        for(j = 0 ; j < w_src ; ++j)
        {
        src1[i*w_src + j] = rand()/double(RAND_MAX);
        src2[i*w_src + j] = rand()/double(RAND_MAX);
    }
    

    // Variables for measuring the execution time
    struct timeval before,after;
    double sbefore, safter, total;

    printf("Image size : %i x %i \n", h_src , w_src);
    
    // Create some matrices holding the image and the resulting FFT
    // The FFTs are complex, we therefore need 2 times the number of pixels in the image
    double * dst_fft_fftw3_src1 = new double[h_src*2*w_src];
    double * dst_fft_fftw3_src2 = new double[h_src*2*w_src];
    double * dst_2fft_fftw3_src1 = new double[h_src*2*w_src];
    double * dst_2fft_fftw3_src2 = new double[h_src*2*w_src];

    // Perform the two 2D FFT with FFTW3
    gettimeofday(&before, NULL);
    for(i = 0 ; i < NB_REPETITIONS ; ++i)
    {
        compute_fft_fftw3(src1, h_src, w_src, dst_fft_fftw3_src1);
        compute_fft_fftw3(src2, h_src, w_src, dst_fft_fftw3_src2);
    }
    gettimeofday(&after, NULL);
    sbefore = before.tv_sec + before.tv_usec * 1E-6;
    safter =after.tv_sec + after.tv_usec * 1E-6;
    total = safter - sbefore;
    printf("Execution time for 2 separate DFT : %e s.\n", total/NB_REPETITIONS);

    // Perform the two 2D FFT with FFTW3
    gettimeofday(&before, NULL);
    for(i = 0 ; i < NB_REPETITIONS ; ++i)
    {
        compute_2fft_fftw3(src1, src2, h_src, w_src, dst_2fft_fftw3_src1, dst_2fft_fftw3_src2);
    }
    gettimeofday(&after, NULL);
    sbefore = before.tv_sec + before.tv_usec * 1E-6;
    safter =after.tv_sec + after.tv_usec * 1E-6;
    total = safter - sbefore;
    printf("Execution time for 2 DFTs at once : %e s.\n", total/NB_REPETITIONS);

    // Compare the outputs
    double error_value;

    error_value = 0.0;
    for(i = 0 ; i < h_src ; ++i)
        for(j = 0 ; j < 2*w_src ; ++j)
            error_value += pow(dst_fft_fftw3_src1[i*2*w_src+j]-dst_2fft_fftw3_src1[i*2*w_src+j],2.0);
    printf("RMS for src1 : %e \n", sqrt(error_value/(double(h_src * 2 * w_src))));

    error_value = 0.0;
    for(i = 0 ; i < h_src ; ++i)
        for(j = 0 ; j < 2*w_src ; ++j)
            error_value += pow(dst_fft_fftw3_src2[i*2*w_src+j]-dst_2fft_fftw3_src2[i*2*w_src+j],2.0);
    printf("RMS for src2 : %e \n", sqrt(error_value/(double(h_src * 2 * w_src))));

    // Free the memory 
    delete[] dst_fft_fftw3_src1;
    delete[] dst_fft_fftw3_src2;
    delete[] dst_2fft_fftw3_src1;
    delete[] dst_2fft_fftw3_src2;

}
