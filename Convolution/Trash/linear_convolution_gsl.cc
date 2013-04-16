// Compile with g++ -o linear_convolution_gsl linear_convolution_gsl.cc `pkg-config --libs --cflags gsl` -lX11 -lpthread -O3 -Wall

#define NDEBUG

#include <iostream>
#include <stdio.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_fft_complex.h>
#include <gsl/gsl_fft_real.h>
#include <gsl/gsl_fft_halfcomplex.h>
#include <math.h>
#include <assert.h>

#include <string>

// To load an image
#include "CImg.h"

// To measure the execution time
#include <sys/time.h>

using namespace cimg_library;

inline double max(double a, double b) { return a > b ? a : b ; }
inline double min(double a, double b) { return a < b ? a : b ; }

/*********************************/
/*       Tool functions          */
/*********************************/

void save_image(const gsl_matrix * image, std::string filename, bool use_log=false)
{
    int i,j;

    double max = gsl_matrix_max(image);
    double min = gsl_matrix_min(image);

    int h_src = image->size1;
    int w_src = image->size2;
    int data[w_src*h_src];

    if(max != min)
    {
        if(!use_log)
        {
            for(i = 0 ; i < h_src ; i ++)
                for(j = 0 ; j < w_src ; j ++)
                {
                data[i*w_src + j] = (int)(255.0*(gsl_matrix_get(image,i,j)-min)/(max-min));
            }
        }
        else
        {
            for(i = 0 ; i < h_src ; i ++)
                for(j = 0 ; j < w_src ; j ++)
                {
                data[i*w_src + j] = (int)(255.0*(log(fabs(gsl_matrix_get(image,i,j))+1)/log(1 + fabs(max))));
            }
        }
    }
    cimg_library::CImg<int> img(data,w_src,h_src,1);
    img.save(filename.c_str());
}

gsl_matrix * setup_filter(std::string filter, int kernel_size = 3)
{
    double * kernel_array;
    int h_kern, w_kern;

    if(filter == "HorizSobel")
    {
        h_kern = 3;
        w_kern = 3;
        kernel_array = new double[h_kern*w_kern];
        kernel_array[0] = -1.0;
        kernel_array[1] = -2.0;
        kernel_array[2] = -1.0;
        kernel_array[3] = 0.0;
        kernel_array[4] = 0.0;
        kernel_array[5] = 0.0;
        kernel_array[6] = 1.0;
        kernel_array[7] = 2.0;
        kernel_array[8] = 1.0;
    }
    else if(filter == "VertSobel")
    {
        h_kern = 3;
        w_kern = 3;
        kernel_array = new double[h_kern*w_kern];
        kernel_array[0] = -1.0;
        kernel_array[1] = 0.0;
        kernel_array[2] = 1.0;
        kernel_array[3] = -2.0;
        kernel_array[4] = 0.0;
        kernel_array[5] = 2.0;
        kernel_array[6] = -1.0;
        kernel_array[7] = 0.0;
        kernel_array[8] = 1.0;
    }
    else if(filter == "HorizPrewitt")
    {
        h_kern = 3;
        w_kern = 3;
        kernel_array = new double[h_kern*w_kern];
        kernel_array[0] = -1.0;
        kernel_array[1] = -1.0;
        kernel_array[2] = -1.0;
        kernel_array[3] = 0.0;
        kernel_array[4] = 0.0;
        kernel_array[5] = 0.0;
        kernel_array[6] = 1.0;
        kernel_array[7] = 1.0;
        kernel_array[8] = 1.0;
    }
    else if(filter == "VertPrewitt")
    {
        h_kern = 3;
        w_kern = 3;
        kernel_array = new double[h_kern*w_kern];
        kernel_array[0] = -1.0;
        kernel_array[1] = 0.0;
        kernel_array[2] = 1.0;
        kernel_array[3] = -1.0;
        kernel_array[4] = 0.0;
        kernel_array[5] = 1.0;
        kernel_array[6] = -1.0;
        kernel_array[7] = 0.0;
        kernel_array[8] = 1.0;
    }
    else if(filter == "Gaussian")
    {
        h_kern = kernel_size;
        w_kern = kernel_size;
        kernel_array = new double[h_kern*w_kern];
        for(int i = 0 ; i < h_kern ; ++i)
            for(int j = 0 ; j < w_kern ; ++j)
                kernel_array[i * w_kern + j] = exp(-(pow(i - h_kern/2,2.0)+pow(j-w_kern/2,2.0))/(2.0 * (h_kern/3.0)*(w_kern/3.0)));
    }
    else
    {
        //printf("Unrecognized filter %s\n", filter);
        return NULL;
    }
    gsl_matrix * kernel = gsl_matrix_alloc(h_kern,w_kern);
    gsl_matrix_memcpy(kernel, &gsl_matrix_view_array(kernel_array, h_kern, w_kern).matrix);
    return kernel;
}

/* ******************************************************************************************************** */
/*          Tool functions to find optimal sizes for the images to convolve with fft and zero-padding       */
/* ******************************************************************************************************** */

int GSL_FACTORS[7] = {7,6 ,5,4,3,2,0}; // end with zero to detect the end of the array

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

/******************************************************/
/* Linear convolution with GSL with an optimal size   */
/******************************************************/

void linear_convolution_fft_gsl(gsl_matrix * src, gsl_matrix * kernel, gsl_matrix * dst)
{
    assert(kernel->size1 <= src->size1 && kernel->size2 <= src->size2 && dst->size1 == src->size1 && dst->size2 == dst->size2);

    int h_src = src->size1;
    int w_src = src->size2;
    int h_kernel = kernel->size1;
    int w_kernel = kernel->size2;

    int height= find_closest_factor(src->size1 + int(kernel->size1+1)/2, GSL_FACTORS);
    int width = find_closest_factor(src->size2 + int(kernel->size2+1)/2, GSL_FACTORS);

    // Create some required matrices
    gsl_fft_complex_workspace * ws_column = gsl_fft_complex_workspace_alloc(height);
    gsl_fft_complex_workspace * ws_line = gsl_fft_complex_workspace_alloc(width);
    gsl_fft_complex_wavetable * wv_column = gsl_fft_complex_wavetable_alloc(height);
    gsl_fft_complex_wavetable * wv_line = gsl_fft_complex_wavetable_alloc(width);

    // We need to create a matrix holding 2 times the number of coefficients of src for the real and imaginary parts
    gsl_matrix * fft = gsl_matrix_alloc(height, 2*width);
    gsl_matrix * fft_copy = gsl_matrix_alloc(height, 2*width);

    int i_src, j_src;
    gsl_matrix_set_zero(fft);
    // memcpy takes care of the strides,
    for(unsigned int i = 0 ; i < src->size1 ; ++i)
    {
        for(unsigned int j = 0 ; j < src->size2 ; ++j)
        {
            fft->data[i*2*width + 2*j] = src->data[i*w_src + j];
        }
    }

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

    // And copy only the real part of fft_src in dst
    for(unsigned int i = 0 ; i < h_src; ++i)
    {
        for(unsigned int j = 0 ; j < w_src ; ++j)
        {
            dst->data[i*w_src + j] = fft_copy->data[i*2*width+2*j];
        }
    }

    gsl_fft_complex_workspace_free(ws_column);
    gsl_fft_complex_workspace_free(ws_line);
    gsl_fft_complex_wavetable_free(wv_column);
    gsl_fft_complex_wavetable_free(wv_line);

    gsl_matrix_free(fft);
    gsl_matrix_free(fft_copy);
}

/******************************************************/
/*                      Main                          */
/******************************************************/

int main(int argc, char * argv[])
{
    if(argc !=3)
    {
        printf("Usage : ./linear_convolution_fftw <image> <kernel_size>\n");
        return -1;
    }

    struct timeval before,after;
    double sbefore, safter, total;

    // To process an image
    char* filename = argv[1];
    int kernel_size = atoi(argv[2]);
    // Read an image file
    CImg<unsigned char> image(filename);
    int w_src, h_src;

    w_src = image.width();
    h_src = image.height();
    gsl_matrix * src = gsl_matrix_alloc(h_src, w_src);
    // src is the image to convolve
    for (int i = 0 ; i < h_src ; i++)
        for (int j = 0 ; j  < w_src ; j++)
            gsl_matrix_set(src,i,j,(int)(image(j,i,0)));

    gsl_matrix * kernel = setup_filter("Gaussian", kernel_size);
    gsl_matrix * dst = gsl_matrix_alloc(h_src, w_src);

    // And compute the linear convolution
    gettimeofday(&before, NULL);
    linear_convolution_fft_gsl(src, kernel, dst);
    gettimeofday(&after, NULL);
    sbefore = before.tv_sec + before.tv_usec * 1E-6;
    safter =after.tv_sec + after.tv_usec * 1E-6;
    total = safter - sbefore;
    std::cout <<"Linear convolution GSL - Total elapsed time: "<<total <<" s \n";

    save_image(src,"src.jpg");
    save_image(kernel, "kernel.jpg");
    save_image(dst,"result.jpg");

    // Free the memory
    gsl_matrix_free(src);
    gsl_matrix_free(kernel);
    gsl_matrix_free(dst);
}
