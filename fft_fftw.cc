// Compute a 2D FFT with FFTW3
// Compile with g++ -o fft_fftw fft_fftw.cc -O3 `pkg-config --libs --cflags fftw3`

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>

// To use FFTW3
#include <fftw3.h>

// To measure execution time
#include <sys/time.h>

/*****************************************************************/
void compute_fft_fftw3(double * src, int height, int width, double * dst)
{
    // Create the required objects for FFTW
    fftw_complex *in_src, *out_src;
    fftw_plan p_forw;

    in_src = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * height * width);
    out_src = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * height * width);

    // Create the plan for performing the forward FFT
    p_forw = fftw_plan_dft_2d(height, width, in_src, out_src, FFTW_FORWARD, FFTW_ESTIMATE);

    // Fill in the real part of the matrix with the image
    for(int i = 0 ; i < height ; ++i)
    {
        for(int j = 0 ; j < width ; ++j)
        {
            in_src[i * width + j][0] = src[i*width + j];
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
            dst[i*2*width + 2*j]= out_src[i * width + j][0];
            dst[i*2*width + 2*j + 1]= out_src[i * width + j][1];
        }
    }

    fftw_destroy_plan(p_forw);
    fftw_free(in_src);
    fftw_free(out_src);
}

int main(int argc, char * argv[])
{
    if(argc != 3)
    {
        std::cerr << " Usage : " << argv[0] << " <width> <height>" << std::endl;
        return 0;
    }

    int i,j;

    int w_src = atoi(argv[1]);
    int h_src = atoi(argv[2]);

    // Variables for measuring the execution time
    struct timeval before,after;
    double sbefore, safter, total;

    // Create the signal of which we compute the FFT
    double * src = new double[h_src*w_src];
    for(i = 0 ; i < h_src ; ++i)
        for(j = 0 ; j < w_src ; ++j)
            src[i*w_src+j] = rand()/double(RAND_MAX);

    printf("Image size : %i x %i \n", h_src , w_src);
    
    // Create some matrices holding the resulting FFT
    // The FFT is complex, we therefore need 2 times the number of pixels in the image
    double * dst_fft_fftw = new double[h_src*2*w_src];

    // Perform the 2D FFT with FFTW3
    gettimeofday(&before, NULL);
    compute_fft_fftw3(src, h_src, w_src, dst_fft_fftw);
    gettimeofday(&after, NULL);
    sbefore = before.tv_sec + before.tv_usec * 1E-6;
    safter =after.tv_sec + after.tv_usec * 1E-6;
    total = safter - sbefore;
    printf("Execution time with FFTW3 : %e s.\n", total);

    // Free the memory
    delete[] dst_fft_fftw;
    delete[] src;
}
