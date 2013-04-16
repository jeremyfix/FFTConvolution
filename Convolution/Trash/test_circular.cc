// Compile with g++ -o circular_convolution_gsl_benchmark circular_convolution_gsl_benchmark.cc `pkg-config --libs --cflags gsl` -O3 -Wall

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

// To measure the execution time
#include <sys/time.h>

#define NB_REPETITIONS 1

inline double max(double a, double b) { return a > b ? a : b ; }
inline double min(double a, double b) { return a < b ? a : b ; }

/* ******************************************************************************************************** */
/*          Tool functions to find optimal sizes for the images to convolve with fft and zero-padding       */
/* ******************************************************************************************************** */

int GSL_FACTORS[7] = {7,6,5,4,3,0}; // end with zero to detect the end of the array

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

    printf("Size : %i x %i \n", height, width);

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

    int height;
    int width;
    //height = find_closest_factor(h_wrapped + int((kernel->size1+1)/2), GSL_FACTORS);
    // width = find_closest_factor(w_wrapped + int((kernel->size2+1)/2), GSL_FACTORS);

    height = find_closest_factor(h_wrapped, GSL_FACTORS);
    width = find_closest_factor(w_wrapped, GSL_FACTORS);

    printf("Size : %i x %i \n", height ,width);

    // Create some required matrices
    gsl_fft_complex_workspace * ws_column = gsl_fft_complex_workspace_alloc(height);
    gsl_fft_complex_workspace * ws_line = gsl_fft_complex_workspace_alloc(width);
    gsl_fft_complex_wavetable * wv_column = gsl_fft_complex_wavetable_alloc(height);
    gsl_fft_complex_wavetable * wv_line = gsl_fft_complex_wavetable_alloc(width);

    // We need to create a matrix holding 2 times the number of coefficients of src for the real and imaginary parts
    gsl_matrix * fft = gsl_matrix_alloc(height, 2*width);
    gsl_matrix * fft_copy = gsl_matrix_alloc(height, 2*width);

    int num_zeros_top = int((height - h_wrapped+1)/2);// This is at least (kernel->size1+1)/2
    int num_zeros_left = int((width - w_wrapped+1)/2);// This is at least (kernel->size2+1)/2

    // Copy and wrap around src
    // fft is filled differently for 10 regions (where the regions filled in with 0 is counted as a single region)
    // 0  |        0          |      0      |       0          | 0
    // 0  | wrap bottom right | wrap bottom | wrap bottom left | 0
    // 0  |     wrap right    |     src     |   wrap left      | 0
    // 0  | wrap top  right   |  wrap top   |  wrap top left   | 0
    // 0  |        0          |      0      |       0          | 0

    gsl_matrix_set_zero(fft);
    // Wrap bottom right
    for(unsigned int j = 0 ; j < int((kernel->size2+1)/2); ++j)
    {
        gsl_vector_memcpy(&gsl_matrix_subcolumn(fft, 2 * (num_zeros_left+j),num_zeros_top,int((kernel->size1+1)/2)).vector,
                          &gsl_matrix_subcolumn(src, src->size2 - int((kernel->size2+1)/2)+j,src->size1-int((kernel->size1+1)/2),int((kernel->size1+1)/2)).vector);
    }

    // Wrap right
    for(unsigned int j = 0 ; j < int((kernel->size2+1)/2); ++j)
    {
        gsl_vector_memcpy(&gsl_matrix_subcolumn(fft, 2 * (num_zeros_left+j),num_zeros_top+int((kernel->size1+1)/2),src->size1).vector,
                          &gsl_matrix_subcolumn(src, src->size2 - int((kernel->size2+1)/2)+j,0,src->size1).vector);
    }
    // Wrap top right
    for(unsigned int j = 0 ; j <  int((kernel->size2+1)/2); ++j)
    {
        gsl_vector_memcpy(&gsl_matrix_subcolumn(fft, 2 * (num_zeros_left+j),num_zeros_top+int((kernel->size1+1)/2)+src->size1,int((kernel->size1)/2)).vector,
                          &gsl_matrix_subcolumn(src, src->size2 - int((kernel->size2+1)/2) + j,0,int((kernel->size1)/2)).vector);
    }
    // Wrap bottom
    for(unsigned int j = 0 ; j < src->size2 ; ++j)
    {
        gsl_vector_memcpy(&gsl_matrix_subcolumn(fft, 2 * (num_zeros_left+int((kernel->size2+1)/2)+j),num_zeros_top,int((kernel->size1+1)/2)).vector,
                          &gsl_matrix_subcolumn(src, j,src->size1-int((kernel->size1+1)/2),int((kernel->size1+1)/2)).vector);
    }
    // Copy the central part
    for(unsigned int j = 0 ; j < src->size2 ; ++j)
    {
        gsl_vector_memcpy(&gsl_matrix_subcolumn(fft, 2 * (num_zeros_left+int((kernel->size2+1)/2)+j),num_zeros_top+int((kernel->size1+1)/2),src->size1).vector,
                          &gsl_matrix_subcolumn(src, j,0,src->size1).vector);
    }
    // Wrap top
    for(unsigned int j = 0 ; j < src->size2 ; ++j)
    {
        gsl_vector_memcpy(&gsl_matrix_subcolumn(fft, 2 * (num_zeros_left+int((kernel->size2+1)/2)+j),num_zeros_top+int((kernel->size1+1)/2)+src->size1,int((kernel->size1)/2)).vector,
                          &gsl_matrix_subcolumn(src, j,0,int((kernel->size1)/2)).vector);
    }
    // Wrap bottom left
    for(unsigned int j = 0 ; j < int((kernel->size2)/2); ++j)
    {
        gsl_vector_memcpy(&gsl_matrix_subcolumn(fft, 2 * (num_zeros_left+int((kernel->size2+1)/2)+src->size2+j),num_zeros_top,int((kernel->size1+1)/2)).vector,
                          &gsl_matrix_subcolumn(src, j,src->size1-int((kernel->size1+1)/2),int((kernel->size1+1)/2)).vector);
    }
    // Wrap left
    for(unsigned int j = 0 ; j < int((kernel->size2)/2); ++j)
    {
        gsl_vector_memcpy(&gsl_matrix_subcolumn(fft, 2 * (num_zeros_left+int((kernel->size2+1)/2)+src->size2+j),num_zeros_top+int((kernel->size1+1)/2),src->size1).vector,
                          &gsl_matrix_subcolumn(src, j,0,src->size1).vector);
    }
    // Wrap top left
    for(unsigned int j = 0 ; j < int((kernel->size2)/2); ++j)
    {
        gsl_vector_memcpy(&gsl_matrix_subcolumn(fft, 2 * (num_zeros_left+int((kernel->size2+1)/2)+src->size2+j),num_zeros_top+int((kernel->size1+1)/2)+src->size1,int(kernel->size1/2)).vector,
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
        gsl_matrix_set_col(dst, j, &gsl_matrix_subcolumn(fft_copy, 2*(num_zeros_left+int((kernel->size2+1)/2)+j),num_zeros_top+int((kernel->size1+1)/2),src->size1).vector);
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
    int h_src_padded = src->size1 + kernel->size1;
    int w_src_padded = src->size2 + kernel->size2;
    int h_filt = kernel->size1;
    int w_filt = kernel->size2;
    double temp;

    // For padding, we build a larger image than src, and copy the right columns/lines to get the wrapping
    gsl_matrix * src_padded = gsl_matrix_alloc(h_src_padded, w_src_padded);
    // There are 9 regions to copy
    // The central part is fed with the original image
    gsl_matrix_memcpy(&gsl_matrix_submatrix(src_padded, h_filt/2, w_filt/2, h_src, w_src).matrix, src);

    // The upper central is the copy of the bottom lines
    gsl_matrix_memcpy(&gsl_matrix_submatrix(src_padded, 0, w_filt/2, h_filt/2, w_src).matrix, &gsl_matrix_submatrix(src, h_src - h_filt/2, 0, h_filt/2, w_src).matrix);
    // The bottom central is the copy of the top lines
    gsl_matrix_memcpy(&gsl_matrix_submatrix(src_padded, h_src + h_filt/2, w_filt/2, h_filt/2, w_src).matrix, &gsl_matrix_submatrix(src, 0, 0, h_filt/2, w_src).matrix);
    // The left central is the copy of the right columns
    gsl_matrix_memcpy(&gsl_matrix_submatrix(src_padded, h_filt/2, 0, h_src, w_filt/2).matrix, &gsl_matrix_submatrix(src, 0, w_src - w_filt/2, h_src, w_filt/2).matrix);
    // The right central is the copy of the left columns
    gsl_matrix_memcpy(&gsl_matrix_submatrix(src_padded, h_filt/2, w_src + w_filt/2 , h_src, w_filt/2).matrix, &gsl_matrix_submatrix(src, 0, 0, h_src, w_filt/2).matrix);

    // The upper left corner is the copy of the bottom right corner of src
    gsl_matrix_memcpy(&gsl_matrix_submatrix(src_padded, 0, 0, h_filt/2, w_filt/2).matrix, &gsl_matrix_submatrix(src, h_src - h_filt/2, w_src - w_filt/2, h_filt/2, w_filt/2).matrix);
    // The upper right corner is the copy of the bottom left corner of src
    gsl_matrix_memcpy(&gsl_matrix_submatrix(src_padded, 0, w_src+w_filt/2, h_filt/2, w_filt/2).matrix, &gsl_matrix_submatrix(src, h_src - h_filt/2, 0, h_filt/2, w_filt/2).matrix);
    // The bottom right corner is the copy of the top left corner of src
    gsl_matrix_memcpy(&gsl_matrix_submatrix(src_padded, h_src + h_filt/2, w_src+w_filt/2, h_filt/2, w_filt/2).matrix, &gsl_matrix_submatrix(src, 0, 0, h_filt/2, w_filt/2).matrix);
    // The bottom left corner is the copy of the top right corner of src
    gsl_matrix_memcpy(&gsl_matrix_submatrix(src_padded, h_src + h_filt/2, 0, h_filt/2, w_filt/2).matrix, &gsl_matrix_submatrix(src, 0, w_src - w_filt/2, h_filt/2, w_filt/2).matrix);

    for (i = h_filt/2 ; i < src->size1 + h_filt/2 ; ++i)
    {
        for (j = w_filt/2 ; j  < src->size2 + w_filt/2 ; ++j)
        {
            temp = 0.0;
            // We browse the kernel
            for (k = 0 ; k < h_filt  ; ++k)
            {
                for(l = 0 ; l < w_filt ; ++l)
                {
                    temp += gsl_matrix_get (src_padded, i + h_filt/2 - k, j + w_filt/2 - l)*gsl_matrix_get (kernel, k , l);
                }
            }
            gsl_matrix_set (dst, i - h_filt/2, j-w_filt/2, temp);
        }
    }

}


/******************************************************/
/*                      Main                          */
/******************************************************/

int main(int argc, char * argv[])
{
    struct timeval before,after;
    double sbefore, safter, total;
    int h_src, w_src;

    h_src = atoi(argv[1]);
    w_src = atoi(argv[1]);

    printf("%i %i ", h_src, w_src);

    gsl_matrix * src = gsl_matrix_alloc(h_src, w_src);
    for(int i = 0 ; i < h_src ; ++i)
        for(int j = 0 ; j < w_src ; ++j)
            gsl_matrix_set(src,i ,j, rand()/double(RAND_MAX));

    gsl_matrix * kernel = gsl_matrix_alloc(h_src, w_src);
    for(int i = 0 ; i < h_src ; ++i)
        for(int j = 0 ; j < w_src ; ++j)
            gsl_matrix_set(kernel,i ,j, rand()/double(RAND_MAX));

    gsl_matrix * dst = gsl_matrix_alloc(h_src, w_src);
    gsl_matrix * dst_optimal = gsl_matrix_alloc(h_src, w_src);
    gsl_matrix * dst_std = gsl_matrix_alloc(h_src, w_src);

    // And compute the linear convolution
    gettimeofday(&before, NULL);
    for(int i = 0 ; i < NB_REPETITIONS ; ++i)
        circular_convolution_fft_gsl(src, kernel, dst);
    gettimeofday(&after, NULL);
    sbefore = before.tv_sec + before.tv_usec * 1E-6;
    safter =after.tv_sec + after.tv_usec * 1E-6;
    total = safter - sbefore;
    printf("%f ", total/NB_REPETITIONS);

    // And compute the linear convolution with an optimal size
    gettimeofday(&before, NULL);
    for(int i = 0 ; i < NB_REPETITIONS ; ++i)
        circular_convolution_fft_gsl_optimal(src, kernel, dst_optimal);
    gettimeofday(&after, NULL);
    sbefore = before.tv_sec + before.tv_usec * 1E-6;
    safter =after.tv_sec + after.tv_usec * 1E-6;
    total = safter - sbefore;
    printf("%f ", total/NB_REPETITIONS);

    // Perform a standard 2D linear convolution
    std_convolution(src, kernel, dst_std);

    // Numerical comparison
    double error_value;
    gsl_vector * diff = gsl_vector_alloc(h_src * w_src);

    gsl_vector_memcpy(diff, &gsl_vector_view_array(dst_std->data, h_src*w_src).vector);
    gsl_vector_sub(diff, &gsl_vector_view_array(dst->data, h_src*w_src).vector);
    error_value = sqrt(gsl_pow_2(gsl_blas_dnrm2(diff))/(double(diff->size)));
    printf("%e ", error_value);

    gsl_vector_memcpy(diff, &gsl_vector_view_array(dst_std->data, h_src*w_src).vector);
    gsl_vector_sub(diff, &gsl_vector_view_array(dst_optimal->data, h_src*w_src).vector);
    error_value = sqrt(gsl_pow_2(gsl_blas_dnrm2(diff))/(double(diff->size)));
    printf("%e ", error_value);

    printf("\n");

    // Free the memory
    gsl_matrix_free(src);
    gsl_matrix_free(kernel);
    gsl_matrix_free(dst);
    gsl_matrix_free(dst_optimal);
    gsl_matrix_free(dst_std);

}
