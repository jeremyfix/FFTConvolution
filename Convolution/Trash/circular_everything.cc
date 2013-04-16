// Compile with g++ -o circular_everything circular_everything.cc `pkg-config --libs --cflags fftw3 gsl` -O3 -Wall

// QUESTION : QUE METTRE DANS LE CODE DE LA GSL ??? Peut etre quand la taille non padde nest pas optimale  alors passer a lautre ? 

// In case you experience issues, comment the following line to enable the assert()
#define NDEBUG

#include <iostream>
#include <stdio.h>
#include <math.h>
#include <assert.h>

//#include <complex.h>
#include <fftw3.h>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_fft_complex.h>
#include <gsl/gsl_fft_real.h>
#include <gsl/gsl_fft_halfcomplex.h>

#include <string>

// To measure the execution time
#include <sys/time.h>

#define NB_REPETITIONS 20

inline double max(double a, double b) { return a > b ? a : b ; }
inline double min(double a, double b) { return a < b ? a : b ; }


/* ******************************************************************************************************** */
/*          Tool functions to find optimal sizes for the images to convolve with fft and zero-padding       */
/* ******************************************************************************************************** */

int FFTW_FACTORS[7] = {13,11,7,5,3,2,0};
int GSL_FACTORS[7] = {7,6,5,4,3,2,0};

// Code taken from gsl/fft/factorize.c
void factorize (const int n,
                int *n_factors,
                int factors[],
                int * implemented_factors)
{
    int nf = 0;
    int ntest = n;
    int factor;
    int i = 0;

    if (n == 0)
    {
        printf("Length n must be positive integer\n");
    }

    if (n == 1)
    {
        factors[0] = 1;
        *n_factors = 1;
        return ;
    }

    /* deal with the implemented factors first */

    while (implemented_factors[i] && ntest != 1)
    {
        factor = implemented_factors[i];
        while ((ntest % factor) == 0)
        {
            ntest = ntest / factor;
            factors[nf] = factor;
            nf++;
        }
        i++;
    }

    // Ok that's it
    if(ntest != 1)
    {
        factors[nf] = ntest;
        nf++;
    }

    /* check that the factorization is correct */
    {
        int product = 1;

        for (i = 0; i < nf; i++)
        {
            product *= factors[i];
        }

        if (product != n)
        {
            printf("factorization failed");
        }
    }

    *n_factors = nf;
}



bool is_optimal(int n, int * implemented_factors)
{
    int nf;
    int factors[64];
    int i = 0;
    factorize(n, &nf, factors,implemented_factors);
    // We just have to check if the last factor belongs to GSL_FACTORS
    while(implemented_factors[i])
    {
        if(factors[nf-1] == implemented_factors[i])
            return true;
        i++;
    }
    return false;
}

int find_closest_factor(int n, int * implemented_factor)
{
    int j;
    if(is_optimal(n,implemented_factor))
    {
        return n;
    }
    else
    {
        j = n+1;
        while(!is_optimal(j,implemented_factor))
            ++j;
        return j;
    }
}

/* ********************************************* */
/*        Circular convolution with GSL,         */
/* ********************************************* */

void gsl_circular_convolution(gsl_matrix * src, gsl_matrix * kernel, gsl_matrix * dst)
{
    // In order to optimze the size for a circular convolution
    // it is sufficient to remind that a circular convolution is the restriction to the central part
    // of a linear convolution performed on a larger image with wrapped around values on the borders

    assert(kernel->size1 <= src->size1 && kernel->size2 <= src->size2 && dst->size1 == src->size1 && dst->size2 == dst->size2);

    int h_src = src->size1;
    int w_src = src->size2;
    int h_kernel = kernel->size1;
    int w_kernel = kernel->size2;

    // The size of the wrapped around image with enough pixel values on both sides (i.e. the size of the kernel)
    int h_wrapped = src->size1 + kernel->size1;
    int w_wrapped = src->size2 + kernel->size2;
    int height = find_closest_factor(h_wrapped, GSL_FACTORS);
    int width = find_closest_factor(w_wrapped, GSL_FACTORS);

    // Create some required matrices
    gsl_fft_complex_workspace * ws_column = gsl_fft_complex_workspace_alloc(height);
    gsl_fft_complex_workspace * ws_line = gsl_fft_complex_workspace_alloc(width);
    gsl_fft_complex_wavetable * wv_column = gsl_fft_complex_wavetable_alloc(height);
    gsl_fft_complex_wavetable * wv_line = gsl_fft_complex_wavetable_alloc(width);

    // We need to create a matrix holding 2 times the number of coefficients of src for the real and imaginary parts
    gsl_matrix * fft = gsl_matrix_alloc(height, 2*width);
    gsl_matrix * fft_copy = gsl_matrix_alloc(height, 2*width);

    // Copy and wrap around src
    // fft is filled differently for 10 regions (where the regions filled in with 0 is counted as a single region)
    // wrap bottom right | wrap bottom | wrap bottom left | 0
    //     wrap right    |     src     |   wrap left      | 0
    // wrap top  right   |  wrap top   |  wrap top left   | 0
    //        0          |      0      |       0          | 0

    gsl_matrix_set_zero(fft);
    int i_src, j_src;
    for(int i = 0 ; i < h_src + h_kernel ; ++i)
    {
        i_src = i - int((h_kernel+1)/2);
        if(i_src < 0)
            i_src += h_src;
        else if(i_src >= h_src)
            i_src -= h_src;
        for(int j = 0 ; j < w_src + w_kernel ; ++j)
        {
            j_src = j - int((w_kernel+1)/2);
            if(j_src < 0)
                j_src += w_src;
            else if(j_src >= w_src)
                j_src -= w_src;

            fft->data[i * 2*width + 2*j] = src->data[i_src * src->size2 + j_src];
        }
    }

    //////
    // Given this new source image, the following is exactly the same as for performing a linear convolution in GSL
    //////

    // when zero-padding, we must ensure that the center of the kernel
    // is copied on the corners of the padded image
    for(int i = 0 ; i < h_kernel ; ++i)
    {
        i_src = i - int(h_kernel/2);
        if(i_src < 0)
            i_src += height;

        for(int j = 0 ; j < w_kernel ; ++j)
        {
            j_src = j - int(w_kernel/2);
            if(j_src < 0)
                j_src += width;
            fft->data[i_src * 2*width + 2*j_src+1] = kernel->data[i * w_kernel + j];
        }
    }

    // We compute the 2 forward DFT at once
    for(int i = 0 ; i < height ; ++i)
    {
        // Apply the FFT on the line i
        gsl_fft_complex_forward (&fft->data[i*2*width],1, width, wv_line, ws_line);
    }
    for(int j = 0 ; j < width ; ++j)
    {
        // Apply the FFT on the column j
        gsl_fft_complex_forward (&fft->data[2*j],width, height, wv_column, ws_column);
    }

    // Their element-wise product : be carefull, the matrices hold complex numbers !
    // We need a copy of fft to perform the product properly
    double re_h, im_h, re_hs, im_hs;
    for(int i = 0 ; i < height ; ++i)
    {
        for(int j = 0 ; j < width ; ++j)
        {
            re_h = fft->data[i*2*width + 2*j];
            im_h = fft->data[i*2*width + 2*j+1];
            re_hs = fft->data[((height-i)%height)*2*width + ((2*width-2*j)%(2*width))];
            im_hs = -fft->data[((height-i)%height)*2*width + ((2*width-2*j)%(2*width))+1];

            fft_copy->data[i*2*width+2*j] = 0.5*(re_h*im_h - re_hs*im_hs);
            fft_copy->data[i*2*width+2*j+1] = -0.25*(re_h*re_h - im_h * im_h - re_hs * re_hs + im_hs * im_hs);
        }
    }

    // And the inverse FFT, which is done in the similar way as before
    for(int i = 0 ; i < height ; ++i)
    {
        // Apply the FFT^{-1} on the line i
        gsl_fft_complex_inverse(&fft_copy->data[i*2*width],1, width, wv_line, ws_line);
    }

    for(int j = 0 ; j < width ; ++j)
    {
        // Apply the FFT^{-1} on the column j
        gsl_fft_complex_inverse(&fft_copy->data[2*j],width, height, wv_column, ws_column);
    }

    // And copy only the real part of the central part of fft_src in dst
    for(unsigned int i = 0 ; i < h_src; ++i)
    {
        for(unsigned int j = 0 ; j < w_src ; ++j)
        {
            dst->data[i*w_src + j] = fft_copy->data[(i+int((h_kernel+1)/2))*2*width+2*(int((w_kernel+1)/2)+j)];
        }
    }

    gsl_fft_complex_workspace_free(ws_column);
    gsl_fft_complex_workspace_free(ws_line);
    gsl_fft_complex_wavetable_free(wv_column);
    gsl_fft_complex_wavetable_free(wv_line);

    gsl_matrix_free(fft);
    gsl_matrix_free(fft_copy);
}

/* ********************************************* */
/*        Circular convolution with FFTW3,         */
/* ********************************************* */

void fftw_circular_convolution(double * src, int h_src, int w_src, double * kernel, int h_kernel, int w_kernel, double * dst)
{
    assert(h_kernel <= h_src && w_kernel <= w_src);

    int height;
    int width;

    height = h_src;
    width = w_src;

    // Create the required objects for FFTW
    // 2 DFTs are computed at the same time
    fftw_complex *in_src, *out_src;
    fftw_plan p_forw;
    fftw_plan p_back;

    in_src = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * height * width);
    out_src = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * height * width);
    p_forw = fftw_plan_dft_2d(height, width, in_src, out_src, FFTW_FORWARD, FFTW_ESTIMATE);
    p_back = fftw_plan_dft_2d(height, width, in_src, out_src, FFTW_BACKWARD, FFTW_ESTIMATE);

    // We need to fill the real part of in_src with the src image

    for(int i = 0 ; i < height ; ++i)
    {
        for(int j = 0 ; j < width ; ++j)
        {
            if( i < h_src && j < w_src)
                in_src[i * width + j][0] = src[i*w_src+j];
            else
                in_src[i * width + j][0] = 0.0;
            in_src[i * width + j][1] = 0.0;
        }
    }

    // We padd and wrap the kernel so that it has the same size as the src image, and that the center of the
    // filter is in (0,0)
    int i_src, j_src;
    for(int i = 0 ; i < h_kernel ; ++i)
    {
        i_src = i - int(h_kernel/2);
        if(i_src < 0)
            i_src += height;

        for(int j = 0 ; j < w_kernel ; ++j)
        {
            j_src = j - int(w_kernel/2);
            if(j_src < 0)
                j_src += width;
            in_src[i_src * width + j_src][1] = kernel[i * w_kernel + j];
        }
    }

    // Compute the forward fft
    fftw_execute(p_forw);

    double re_h, im_h, re_hs, im_hs;
    // Compute the element-wise product
    for(int i = 0 ; i < height ; ++i)
    {
        for(int j = 0 ; j < width ; ++j)
        {
            re_h = out_src[i*width+ j][0];
            im_h = out_src[i*width+ j][1];
            re_hs = out_src[((height-i)%height)*width + (width-j)%width][0];
            im_hs = - out_src[((height-i)%height)*width + (width-j)%width][1];

            in_src[i*width+j][0] =  0.5*(re_h*im_h - re_hs*im_hs);
            in_src[i*width+j][1] = -0.25*(re_h*re_h - im_h * im_h - re_hs * re_hs + im_hs * im_hs);
        }
    }

    // Compute the backward fft
    fftw_execute(p_back);

    // Now we just need to copy the right part of in_src into dst
    for(int i  = 0 ; i < h_src ; ++i)
    {
        for(int j = 0 ; j < w_src ; ++j)
        {
            dst[i*w_src+ j] = out_src[i * width + j][0]/ double(width * height);
        }
    }

    fftw_destroy_plan(p_forw);
    fftw_destroy_plan(p_back);
    fftw_free(in_src); fftw_free(out_src);
}


/******************************************************/
/* Standard 2D linear convolution                     */
/******************************************************/
void std_circular_convolution(double * src, int h_src, int w_src, double * kernel, int h_kernel, int w_kernel, double * dst)
{
    int i,j,k,l;
    double temp;

    int i_src, j_src;
    for (i = 0 ; i < h_src ; ++i)
    {
        for (j = 0 ; j  < w_src ; ++j)
        {
            temp = 0.0;
            // We browse the kernel
            for (k = 0 ; k < h_kernel  ; ++k)
            {
                i_src = i + h_kernel/2 - k;
                if(i_src < 0)
                    i_src += h_src;
                else if(i_src >= h_src)
                    i_src -= h_src;

                for(l = 0 ; l < w_kernel ; ++l)
                {
                    j_src = j + w_kernel/2 - l;
                    if(j_src < 0)
                        j_src += w_src;
                    else if(j_src >= w_src)
                        j_src -= w_src;

                    temp += src[i_src*w_src + j_src]* kernel[k*w_kernel+l];
                }
            }
            dst[i*w_src+j] = temp;
        }
    }
}

/* ******************************************************************************************************** */
/*                   Main : it loads an image and convolves it with a gaussian filter                       */
/* ******************************************************************************************************** */

int main(int argc, char * argv[])
{
    if(argc !=3)
    {
        printf("Usage : ./linear_convolution_fftw <image_size> <kernel_size>\n");
        return -1;
    }

    struct timeval before,after;
    double sbefore, safter, total;

    // To process an image
    int img_size = atoi(argv[1]);
    int kernel_size = atoi(argv[2]);

    int w_src, h_src;
    int h_kernel, w_kernel;

    w_src = img_size;
    h_src = img_size;

    h_kernel = kernel_size;
    w_kernel = kernel_size;

    printf("%i %i ", img_size, kernel_size);

    // *********** FFTW ******************//
    // Arrays used for the FFTW convolution
    double * src = new double[h_src* w_src];
    for (int i = 0 ; i < h_src ; i++)
        for (int j = 0 ; j  < w_src ; j++)
            src[i*w_src+j] = rand() / (double(RAND_MAX));

    double * kernel = new double[h_kernel* w_kernel];
    for (int i = 0 ; i < h_kernel ; i++)
        for (int j = 0 ; j  < w_kernel ; j++)
            kernel[i*w_kernel+j] = rand() / (double(RAND_MAX));

    double * dst = new double[h_src* w_src];
    for (int i = 0 ; i < h_src ; i++)
        for (int j = 0 ; j  < w_src ; j++)
            dst[i*w_src+j] = 0.0;

    // And compute the linear convolution with an optimal size
    gettimeofday(&before, NULL);
    for(int i = 0 ; i < NB_REPETITIONS; ++i)
        fftw_circular_convolution(src, h_src, w_src,kernel, h_kernel, w_kernel, dst);
    gettimeofday(&after, NULL);
    sbefore = before.tv_sec + before.tv_usec * 1E-6;
    safter =after.tv_sec + after.tv_usec * 1E-6;
    total = safter - sbefore;
    printf("%f ", total/double(NB_REPETITIONS));
    //printf("Linear convolution FFTW - Total elapsed time: %f s.\n", total/double(NB_REPETITIONS));

    // Free the memory
    delete[] src;
    delete[] kernel;
    delete[] dst;

    // *********** GSL ******************//
    gsl_matrix * src_gsl = gsl_matrix_alloc(h_src, w_src);
    for (int i = 0 ; i < h_src ; i++)
        for (int j = 0 ; j  < w_src ; j++)
            gsl_matrix_set(src_gsl,i,j, rand() / (double(RAND_MAX)));

    gsl_matrix * kernel_gsl = gsl_matrix_alloc(h_kernel, w_kernel);
    for (int i = 0 ; i < h_kernel ; i++)
        for (int j = 0 ; j  < w_kernel ; j++)
            gsl_matrix_set(kernel_gsl, i,j,rand() / (double(RAND_MAX)));

    gsl_matrix * dst_gsl = gsl_matrix_alloc(h_src, w_src);
    gsl_matrix_set_zero(dst_gsl);

    // And compute the linear convolution with an optimal size
    gettimeofday(&before, NULL);
    for(int i = 0 ; i < NB_REPETITIONS; ++i)
        gsl_circular_convolution(src_gsl, kernel_gsl, dst_gsl);
    gettimeofday(&after, NULL);
    sbefore = before.tv_sec + before.tv_usec * 1E-6;
    safter =after.tv_sec + after.tv_usec * 1E-6;
    total = safter - sbefore;
    printf("%f ", total/double(NB_REPETITIONS));
    //printf("Linear convolution GSL - Total elapsed time: %f s.\n", total/double(NB_REPETITIONS));

    gsl_matrix_free(src_gsl);
    gsl_matrix_free(kernel_gsl);
    gsl_matrix_free(dst_gsl);

    // *********** Standard ******************//

    src = new double[h_src* w_src];
    for (int i = 0 ; i < h_src ; i++)
        for (int j = 0 ; j  < w_src ; j++)
            src[i*w_src+j] = rand() / (double(RAND_MAX));

    kernel = new double[h_kernel* w_kernel];
    for (int i = 0 ; i < h_kernel ; i++)
        for (int j = 0 ; j  < w_kernel ; j++)
            kernel[i*w_kernel+j] = rand() / (double(RAND_MAX));

    dst = new double[h_src* w_src];
    for (int i = 0 ; i < h_src ; i++)
        for (int j = 0 ; j  < w_src ; j++)
            dst[i*w_src+j] = 0.0;

    // And compute the linear convolution with an optimal size
    gettimeofday(&before, NULL);
    //for(int i = 0 ; i < NB_REPETITIONS; ++i)
    //  std_circular_convolution(src, h_src, w_src,kernel, h_kernel, w_kernel, dst);
    gettimeofday(&after, NULL);
    sbefore = before.tv_sec + before.tv_usec * 1E-6;
    safter =after.tv_sec + after.tv_usec * 1E-6;
    total = safter - sbefore;
    printf("%f ", total/double(NB_REPETITIONS));
    //printf("Linear convolution Std - Total elapsed time: %f s.\n", total/double(NB_REPETITIONS));

    // Free the memory
    delete [] src;
    delete [] kernel;
    delete [] dst;

    printf("\n");
}

