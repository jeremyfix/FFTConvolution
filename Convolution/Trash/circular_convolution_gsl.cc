// Compile with g++ -o circular_convolution_gsl circular_convolution_gsl.cc `pkg-config --libs --cflags gsl` -lX11 -lpthread -O3 -Wall

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
    //double datad[w_src*h_src];

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

gsl_matrix * setup_filter(std::string filter)
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
    else if(filter == "Test")
    {
        h_kern = 15;
        w_kern = 15;
        kernel_array = new double[h_kern*w_kern];
        for(int i = 0 ; i < h_kern ; ++i)
            for(int j = 0 ; j < w_kern; ++j)
                kernel_array[i*w_kern + j] = rand() / double(RAND_MAX);
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
        h_kern = 11;
        w_kern = 11;
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

/*********************************/
/* Circular convolution with GSL   */
/*********************************/

void circular_convolution_fft_gsl(gsl_matrix * src, gsl_matrix * kernel, gsl_matrix * dst)
{   
    assert(kernel->size1 <= src->size1 && kernel->size2 <= src->size2 && dst->size1 == src->size1 && dst->size2 == dst->size2);

    int height;
    int width;

    height = src->size1;
    width = src->size2;

    // Create some required matrices
    gsl_fft_complex_workspace * ws_column = gsl_fft_complex_workspace_alloc(height);
    gsl_fft_complex_workspace * ws_line = gsl_fft_complex_workspace_alloc(width);
    gsl_fft_complex_wavetable * wv_column = gsl_fft_complex_wavetable_alloc(height);
    gsl_fft_complex_wavetable * wv_line = gsl_fft_complex_wavetable_alloc(width);

    // We need to create a matrix holding 2 times the number of coefficients of src for the real and imaginary parts
    gsl_matrix * fft = gsl_matrix_alloc(height, 2*width);
    gsl_matrix * fft_copy = gsl_matrix_alloc(height, 2*width);

    gsl_matrix_set_zero(fft);
    // memcpy takes care of the strides,
    for(unsigned int i = 0 ; i < src->size2 ; ++i)
    {
        gsl_vector_memcpy(&gsl_matrix_subcolumn(fft,2*i,0, src->size1).vector, &gsl_matrix_column(src, i).vector);
    }

    // when zero-padding, we must ensure that the center of the kernel
    // is copied on the corners of the padded image
    // There are four quadrants to copy

    // The columns on the left of the center are put on the extreme right of fft_kernel
    for(unsigned int i = 0 ; i < int(kernel->size2/2) ; ++i)
    {
        // The top rows of kernel are put on the bottom rows of fft_kernel
        gsl_vector_memcpy(&gsl_matrix_subcolumn(fft,2*width-2*int(kernel->size2/2)+2*i+1,height - int(kernel->size1/2),int(kernel->size1/2)).vector,&gsl_matrix_subcolumn(kernel, i, 0, int(kernel->size1/2)).vector);
        // The bottom rows of the kernel are put on the top rows of fft_kernel
        gsl_vector_memcpy(&gsl_matrix_subcolumn(fft,2*width-2*int(kernel->size2/2)+2*i+1,0,int((kernel->size1+1)/2)).vector,&gsl_matrix_subcolumn(kernel, i, int(kernel->size1/2), int((kernel->size1+1)/2) ).vector);
    }
    // The columns on the right of the center are put on the extreme left of fft_kernel
    for(unsigned int i = int(kernel->size2/2) ; i < kernel->size2 ; ++i)
    {
        // The top rows of kernel are put on the bottom rows of fft_kernel
        gsl_vector_memcpy(&gsl_matrix_subcolumn(fft,2*(i-int(kernel->size2/2))+1,height - int(kernel->size1/2),int(kernel->size1/2)).vector,&gsl_matrix_subcolumn(kernel, i, 0, int(kernel->size1/2)).vector);
        // The bottom rows of the kernel are put on the top rows of fft_kernel
        gsl_vector_memcpy(&gsl_matrix_subcolumn(fft,2*(i-int(kernel->size2/2))+1,0,int((kernel->size1+1)/2)).vector,&gsl_matrix_subcolumn(kernel, i, int(kernel->size1/2), int((kernel->size1+1)/2)).vector);
    }

    //save_image(fft, "image.png",true);

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
            re_h = gsl_matrix_get(fft,i, 2*j);
            im_h = gsl_matrix_get(fft,i, 2*j+1);
            re_hs = gsl_matrix_get(fft,(height-i)%height, ((2*width-2*j)%(2*width)));
            im_hs = - gsl_matrix_get(fft,(height-i)%height, ((2*width-2*j)%(2*width))+1);

            gsl_matrix_set(fft_copy, i, 2*j, 0.5*(re_h*im_h - re_hs*im_hs));
            gsl_matrix_set(fft_copy, i, 2*j+1, -0.25*(re_h*re_h - im_h * im_h - re_hs * re_hs + im_hs * im_hs));
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
    for(unsigned int i = 0 ; i < src->size2 ; ++i)
    {
        gsl_matrix_set_col(dst, i, &gsl_matrix_subcolumn(fft_copy, 2*i,0,src->size1).vector);
    }

    gsl_fft_complex_workspace_free(ws_column);
    gsl_fft_complex_workspace_free(ws_line);
    gsl_fft_complex_wavetable_free(wv_column);
    gsl_fft_complex_wavetable_free(wv_line);

    gsl_matrix_free(fft);
    gsl_matrix_free(fft_copy);
}

/******************************************************/
/* Circular convolution with GSL with an optimal size   */
/******************************************************/

void circular_convolution_fft_gsl_optimal(gsl_matrix * src, gsl_matrix * kernel, gsl_matrix * dst)
{
    // In order to optimze the size for a circular convolution
    // it is sufficient to remind that a circular convolution is the restriction to the central part
    // of a linear convolution performed on a larger image with wrapped around values on the borders

    assert(kernel->size1 <= src->size1 && kernel->size2 <= src->size2 && dst->size1 == src->size1 && dst->size2 == dst->size2);

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
    // Wrap bottom right
    for(unsigned int j = 0 ; j < int((kernel->size2+1)/2); ++j)
    {
        gsl_vector_memcpy(&gsl_matrix_subcolumn(fft, 2 * j,0,int((kernel->size1+1)/2)).vector,
                          &gsl_matrix_subcolumn(src, src->size2 - int((kernel->size2+1)/2)+j,src->size1-int((kernel->size1+1)/2),int((kernel->size1+1)/2)).vector);
    }

    // Wrap right
    for(unsigned int j = 0 ; j < int((kernel->size2+1)/2); ++j)
    {
        gsl_vector_memcpy(&gsl_matrix_subcolumn(fft, 2 * j,int((kernel->size1+1)/2),src->size1).vector,
                          &gsl_matrix_subcolumn(src, src->size2 - int((kernel->size2+1)/2)+j,0,src->size1).vector);
    }
    // Wrap top right
    for(unsigned int j = 0 ; j <  int((kernel->size2+1)/2); ++j)
    {
        gsl_vector_memcpy(&gsl_matrix_subcolumn(fft, 2 * j,int((kernel->size1+1)/2)+src->size1,int((kernel->size1)/2)).vector,
                          &gsl_matrix_subcolumn(src, src->size2 - int((kernel->size2+1)/2) + j,0,int((kernel->size1)/2)).vector);
    }
    // Wrap bottom
    for(unsigned int j = 0 ; j < src->size2 ; ++j)
    {
        gsl_vector_memcpy(&gsl_matrix_subcolumn(fft, 2 * (int((kernel->size2+1)/2)+j),0,int((kernel->size1+1)/2)).vector,
                          &gsl_matrix_subcolumn(src, j,src->size1-int((kernel->size1+1)/2),int((kernel->size1+1)/2)).vector);
    }
    // Copy the central part
    for(unsigned int j = 0 ; j < src->size2 ; ++j)
    {
        gsl_vector_memcpy(&gsl_matrix_subcolumn(fft, 2 * (int((kernel->size2+1)/2)+j),int((kernel->size1+1)/2),src->size1).vector,
                          &gsl_matrix_subcolumn(src, j,0,src->size1).vector);
    }
    // Wrap top
    for(unsigned int j = 0 ; j < src->size2 ; ++j)
    {
        gsl_vector_memcpy(&gsl_matrix_subcolumn(fft, 2 * (int((kernel->size2+1)/2)+j),int((kernel->size1+1)/2)+src->size1,int((kernel->size1)/2)).vector,
                          &gsl_matrix_subcolumn(src, j,0,int((kernel->size1)/2)).vector);
    }
    // Wrap bottom left
    for(unsigned int j = 0 ; j < int((kernel->size2)/2); ++j)
    {
        gsl_vector_memcpy(&gsl_matrix_subcolumn(fft, 2 * (int((kernel->size2+1)/2)+src->size2+j),0,int((kernel->size1+1)/2)).vector,
                          &gsl_matrix_subcolumn(src, j,src->size1-int((kernel->size1+1)/2),int((kernel->size1+1)/2)).vector);
    }
    // Wrap left
    for(unsigned int j = 0 ; j < int((kernel->size2)/2); ++j)
    {
        gsl_vector_memcpy(&gsl_matrix_subcolumn(fft, 2 * (int((kernel->size2+1)/2)+src->size2+j),int((kernel->size1+1)/2),src->size1).vector,
                          &gsl_matrix_subcolumn(src, j,0,src->size1).vector);
    }
    // Wrap top left
    for(unsigned int j = 0 ; j < int((kernel->size2)/2); ++j)
    {
        gsl_vector_memcpy(&gsl_matrix_subcolumn(fft, 2 * (int((kernel->size2+1)/2)+src->size2+j),int((kernel->size1+1)/2)+src->size1,int(kernel->size1/2)).vector,
                          &gsl_matrix_subcolumn(src, j,0,int(kernel->size1/2)).vector);
    }
    //save_image(fft,"wrapped_src.jpg");

    //////
    // Given this new source image, the following is exactly the same as for performing a linear convolution in GSL
    //////

    // when zero-padding, we must ensure that the center of the kernel
    // is copied on the corners of the padded image
    // There are four quadrants to copy

    // The columns on the left of the center are put on the extreme right of fft_kernel
    for(unsigned int i = 0 ; i < int(kernel->size2/2) ; ++i)
    {
        // The top rows of kernel are put on the bottom rows of fft_kernel
        gsl_vector_memcpy(&gsl_matrix_subcolumn(fft,2*width-2*int(kernel->size2/2)+2*i+1,height - int(kernel->size1/2),int(kernel->size1/2)).vector,&gsl_matrix_subcolumn(kernel, i, 0, int(kernel->size1/2)).vector);
        // The bottom rows of the kernel are put on the top rows of fft_kernel
        gsl_vector_memcpy(&gsl_matrix_subcolumn(fft,2*width-2*int(kernel->size2/2)+2*i+1,0,int((kernel->size1+1)/2)).vector,&gsl_matrix_subcolumn(kernel, i, int(kernel->size1/2), int((kernel->size1+1)/2) ).vector);
    }
    // The columns on the right of the center are put on the extreme left of fft_kernel
    for(unsigned int i = int(kernel->size2/2) ; i < kernel->size2 ; ++i)
    {
        // The top rows of kernel are put on the bottom rows of fft_kernel
        gsl_vector_memcpy(&gsl_matrix_subcolumn(fft,2*(i-int(kernel->size2/2))+1,height - int(kernel->size1/2),int(kernel->size1/2)).vector,&gsl_matrix_subcolumn(kernel, i, 0, int(kernel->size1/2)).vector);
        // The bottom rows of the kernel are put on the top rows of fft_kernel
        gsl_vector_memcpy(&gsl_matrix_subcolumn(fft,2*(i-int(kernel->size2/2))+1,0,int((kernel->size1+1)/2)).vector,&gsl_matrix_subcolumn(kernel, i, int(kernel->size1/2), int((kernel->size1+1)/2)).vector);
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
            re_h = gsl_matrix_get(fft,i, 2*j);
            im_h = gsl_matrix_get(fft,i, 2*j+1);
            re_hs = gsl_matrix_get(fft,(height-i)%height, ((2*width-2*j)%(2*width)));
            im_hs = - gsl_matrix_get(fft,(height-i)%height, ((2*width-2*j)%(2*width))+1);

            gsl_matrix_set(fft_copy, i, 2*j, 0.5*(re_h*im_h - re_hs*im_hs));
            gsl_matrix_set(fft_copy, i, 2*j+1, -0.25*(re_h*re_h - im_h * im_h - re_hs * re_hs + im_hs * im_hs));
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
    for(unsigned int j = 0 ; j < src->size2 ; ++j)
    {
        gsl_matrix_set_col(dst, j, &gsl_matrix_subcolumn(fft_copy, 2*(int((kernel->size2+1)/2)+j),int((kernel->size1+1)/2),src->size1).vector);
    }

    gsl_fft_complex_workspace_free(ws_column);
    gsl_fft_complex_workspace_free(ws_line);
    gsl_fft_complex_wavetable_free(wv_column);
    gsl_fft_complex_wavetable_free(wv_line);

    gsl_matrix_free(fft);
    gsl_matrix_free(fft_copy);
}

/******************************************************/
/* Standard 2D convolution for numerical comparison   */
/******************************************************/

void std_convolution(gsl_matrix * src, gsl_matrix * kernel, gsl_matrix * dst)
{
    int i,j,k,l;
    int h_src = src->size1;
    int w_src = src->size2;
    int h_filt = kernel->size1;
    int w_filt = kernel->size2;
    double temp;

    int i_src, j_src;
    for (i = 0 ; i < h_src ; ++i)
    {
        for (j = 0 ; j  < w_src ; ++j)
        {
            temp = 0.0;
            // We browse the kernel
            for (k = 0 ; k < h_filt  ; ++k)
            {
                i_src = i + h_filt/2 - k;
                if(i_src < 0)
                    i_src += h_src;
                else if(i_src >= h_src)
                    i_src -= h_src;
                for(l = 0 ; l < w_filt ; ++l)
                {
                    j_src = j + w_filt/2 - l;
                    if(j_src < 0)
                        j_src += w_src;
                    else if(j_src >= w_src)
                        j_src -= w_src;

                    temp += src->data[i_src * w_src + j_src] * kernel->data[k * w_filt + l];
                    //temp += gsl_matrix_get (src, i_src, j_src)*gsl_matrix_get (kernel, k , l);
                }
            }
            dst->data[i*w_src +j] = temp;
            //gsl_matrix_set (dst, i, j, temp);
        }
    }
}


/******************************************************/
/*                      Main                          */
/******************************************************/

int main(int argc, char * argv[])
{
    if(argc !=2)
    {
        printf("Usage : ./circular_convolution_gsl <image>\n");
        return -1;
    }

    struct timeval before,after;
    double sbefore, safter, total;

    // To process an image
    char* filename = argv[1];
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

    gsl_matrix * kernel = setup_filter("Test");
    gsl_matrix * dst = gsl_matrix_alloc(h_src, w_src);
    gsl_matrix * dst_optimal = gsl_matrix_alloc(h_src, w_src);
    gsl_matrix * dst_std = gsl_matrix_alloc(h_src, w_src);

    // And compute the circular convolution
    gettimeofday(&before, NULL);
    circular_convolution_fft_gsl(src, kernel, dst);
    gettimeofday(&after, NULL);
    sbefore = before.tv_sec + before.tv_usec * 1E-6;
    safter =after.tv_sec + after.tv_usec * 1E-6;
    total = safter - sbefore;
    std::cout <<"Circular convolution GSL - Total elapsed time: "<<total <<" s \n";

    // And compute the circular convolution with an optimal size
    gettimeofday(&before, NULL);
    circular_convolution_fft_gsl_optimal(src, kernel, dst_optimal);
    gettimeofday(&after, NULL);
    sbefore = before.tv_sec + before.tv_usec * 1E-6;
    safter =after.tv_sec + after.tv_usec * 1E-6;
    total = safter - sbefore;
    std::cout <<"Circular convolution GSL, optimal size - Total elapsed time: "<<total <<" s \n";

    save_image(src,"src.jpg");
    save_image(kernel, "kernel.jpg");
    save_image(dst,"result.jpg");
    save_image(dst_optimal,"result_optim.jpg");

    // Perform a standard 2D circular convolution
    gettimeofday(&before, NULL);
    std_convolution(src, kernel, dst_std);
    gettimeofday(&after, NULL);
    sbefore = before.tv_sec + before.tv_usec * 1E-6;
    safter =after.tv_sec + after.tv_usec * 1E-6;
    total = safter - sbefore;
    std::cout <<"Standard circular convolution  - Total elapsed time: "<<total <<" s \n";
    save_image(dst_std, "result_std.jpg");

    // Numerical comparison
    double error_value;
    gsl_vector * diff = gsl_vector_alloc(h_src * w_src);

    gsl_vector_memcpy(diff, &gsl_vector_view_array(dst_std->data, h_src*w_src).vector);
    gsl_vector_sub(diff, &gsl_vector_view_array(dst->data, h_src*w_src).vector);
    error_value = sqrt(gsl_pow_2(gsl_blas_dnrm2(diff))/(double(diff->size)));
    std::cout << " RMS (std, gsl) : " << std::scientific << error_value << std::endl;

    gsl_vector_memcpy(diff, &gsl_vector_view_array(dst_std->data, h_src*w_src).vector);
    gsl_vector_sub(diff, &gsl_vector_view_array(dst_optimal->data, h_src*w_src).vector);
    error_value = sqrt(gsl_pow_2(gsl_blas_dnrm2(diff))/(double(diff->size)));
    std::cout << " RMS (std, gsl optimal) : " << std::scientific << error_value << std::endl;

    // Free the memory
    gsl_matrix_free(src);
    gsl_matrix_free(kernel);
    gsl_matrix_free(dst);
    gsl_matrix_free(dst_optimal);
    gsl_matrix_free(dst_std);
}
