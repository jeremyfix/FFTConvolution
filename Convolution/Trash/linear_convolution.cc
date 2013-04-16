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

#define NB_REPETITIONS 30

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
/* Linear convolution with GSL   */
/*********************************/

void linear_convolution_fft_gsl(gsl_matrix * src, gsl_matrix * kernel, gsl_matrix * dst)
{
    assert(kernel->size1 <= src->size1 && kernel->size2 <= src->size2 && dst->size1 == src->size1 && dst->size2 == dst->size2);

    int height;
    int width;

    height = src->size1 + int(kernel->size1+1)/2;
    width = src->size2 + int(kernel->size2+1)/2;

    // Create some required matrices
    gsl_fft_complex_workspace * ws_column = gsl_fft_complex_workspace_alloc(height);
    gsl_fft_complex_workspace * ws_line = gsl_fft_complex_workspace_alloc(width);
    gsl_fft_complex_wavetable * wv_column = gsl_fft_complex_wavetable_alloc(height);
    gsl_fft_complex_wavetable * wv_line = gsl_fft_complex_wavetable_alloc(width);

    // We need to create a matrix holding 2 times the number of coefficients of src for the real and imaginary parts
    gsl_matrix * fft_src = gsl_matrix_alloc(height, 2*width);
    gsl_matrix_set_zero(fft_src);
    // memcpy takes care of the strides,
    for(unsigned int i = 0 ; i < src->size2 ; ++i)
    {
        gsl_vector_memcpy(&gsl_matrix_subcolumn(fft_src,2*i,0, src->size1).vector, &gsl_matrix_column(src, i).vector);
    }

    // We padd the kernel so that it has the same size as the src image
    // we consider that the kernel is smaller than src
    gsl_matrix * fft_kernel = gsl_matrix_alloc(height, 2*width);
    gsl_matrix_set_zero(fft_kernel);

    // when zero-padding, we must ensure that the center of the kernel
    // is copied on the corners of the padded image
    // There are four quadrants to copy

    // The columns on the left of the center are put on the extreme right of fft_kernel
    for(unsigned int i = 0 ; i < int(kernel->size2/2) ; ++i)
    {
        // The top rows of kernel are put on the bottom rows of fft_kernel
        gsl_vector_memcpy(&gsl_matrix_subcolumn(fft_kernel,fft_kernel->size2-2*int(kernel->size2/2)+2*i,fft_kernel->size1 - int(kernel->size1/2),int(kernel->size1/2)).vector,&gsl_matrix_subcolumn(kernel, i, 0, int(kernel->size1/2)).vector);
        // The bottom rows of the kernel are put on the top rows of fft_kernel
        gsl_vector_memcpy(&gsl_matrix_subcolumn(fft_kernel,fft_kernel->size2-2*int(kernel->size2/2)+2*i,0,int((kernel->size1+1)/2)).vector,&gsl_matrix_subcolumn(kernel, i, int(kernel->size1/2), int((kernel->size1+1)/2) ).vector);
    }
    // The columns on the right of the center are put on the extreme left of fft_kernel
    for(unsigned int i = int(kernel->size2/2) ; i < kernel->size2 ; ++i)
    {
        // The top rows of kernel are put on the bottom rows of fft_kernel
        gsl_vector_memcpy(&gsl_matrix_subcolumn(fft_kernel,2*(i-int(kernel->size2/2)),fft_kernel->size1 - int(kernel->size1/2),int(kernel->size1/2)).vector,&gsl_matrix_subcolumn(kernel, i, 0, int(kernel->size1/2)).vector);
        // The bottom rows of the kernel are put on the top rows of fft_kernel
        gsl_vector_memcpy(&gsl_matrix_subcolumn(fft_kernel,2*(i-int(kernel->size2/2)),0,int((kernel->size1+1)/2)).vector,&gsl_matrix_subcolumn(kernel, i, int(kernel->size1/2), int((kernel->size1+1)/2)).vector);
    }

    // We need to compute a 2D fft, which is performed by performing several 1D FFT
    // 1 - Apply the FFT on the lines
    // 2 - Apply the FFT on the columns of the previous result
    for(int i = 0 ; i < height ; ++i)
    {
        // Apply the FFT on the line i
        gsl_fft_complex_forward (&fft_src->data[i*2*width],1, width, wv_line, ws_line);
        gsl_fft_complex_forward (&fft_kernel->data[i*2*width],1, width, wv_line, ws_line);
    }
    for(int j = 0 ; j < width ; ++j)
    {
        // Apply the FFT on the column j
        gsl_fft_complex_forward (&fft_src->data[2*j],width, height, wv_column, ws_column);
        gsl_fft_complex_forward (&fft_kernel->data[2*j],width, height, wv_column, ws_column);
    }

    // Their element-wise product : be carefull, the matrices hold complex numbers !
    double rea, ima, reb, imb;
    for(int i = 0 ; i < height ; ++i)
    {
        for(int j = 0 ; j < width ; ++j)
        {
            rea = gsl_matrix_get(fft_src, i, 2*j);
            ima = gsl_matrix_get(fft_src, i, 2*j+1);
            reb = gsl_matrix_get(fft_kernel, i, 2*j);
            imb = gsl_matrix_get(fft_kernel, i, 2*j+1);
            gsl_matrix_set(fft_src, i, 2*j, rea * reb - ima * imb);
            gsl_matrix_set(fft_src, i, 2*j+1, rea * imb + ima * reb);
        }
    }

    // And the inverse FFT, which is done in the similar way as before
    for(int i = 0 ; i < height ; ++i)
    {
        // Apply the FFT^{-1} on the line i
        gsl_fft_complex_inverse(&fft_src->data[i*2*width],1, width, wv_line, ws_line);
    }

    for(int j = 0 ; j < width ; ++j)
    {
        // Apply the FFT^{-1} on the column j
        gsl_fft_complex_inverse(&fft_src->data[2*j],width, height, wv_column, ws_column);
    }

    // And copy only the real part of fft_src in dst
    for(unsigned int i = 0 ; i < src->size2 ; ++i)
    {
        gsl_matrix_set_col(dst, i, &gsl_matrix_subcolumn(fft_src, 2*i,0,src->size1).vector);
    }

    gsl_fft_complex_workspace_free(ws_column);
    gsl_fft_complex_workspace_free(ws_line);
    gsl_fft_complex_wavetable_free(wv_column);
    gsl_fft_complex_wavetable_free(wv_line);

    gsl_matrix_free(fft_src);
    gsl_matrix_free(fft_kernel);
}

/******************************************************/
/* Linear convolution with GSL with an optimal size   */
/******************************************************/

void linear_convolution_fft_gsl_optimal(gsl_matrix * src, gsl_matrix * kernel, gsl_matrix * dst)
{
    assert(kernel->size1 <= src->size1 && kernel->size2 <= src->size2 && dst->size1 == src->size1 && dst->size2 == dst->size2);

    int height;
    int width;

    height = find_closest_factor(src->size1 + int(kernel->size1+1)/2, GSL_FACTORS);
    width = find_closest_factor(src->size2 + int(kernel->size2+1)/2, GSL_FACTORS);
    //printf("Size : %i %i \n", height, width);

    // Create some required matrices
    gsl_fft_complex_workspace * ws_column = gsl_fft_complex_workspace_alloc(height);
    gsl_fft_complex_workspace * ws_line = gsl_fft_complex_workspace_alloc(width);
    gsl_fft_complex_wavetable * wv_column = gsl_fft_complex_wavetable_alloc(height);
    gsl_fft_complex_wavetable * wv_line = gsl_fft_complex_wavetable_alloc(width);

    // We need to create a matrix holding 2 times the number of coefficients of src for the real and imaginary parts
    gsl_matrix * fft_src = gsl_matrix_alloc(height, 2*width);
    gsl_matrix_set_zero(fft_src);
    // memcpy takes care of the strides,
    for(unsigned int i = 0 ; i < src->size2 ; ++i)
    {
        gsl_vector_memcpy(&gsl_matrix_subcolumn(fft_src,2*i,0, src->size1).vector, &gsl_matrix_column(src, i).vector);
    }

    // We padd the kernel so that it has the same size as the src image
    // we consider that the kernel is smaller than src
    gsl_matrix * fft_kernel = gsl_matrix_alloc(height, 2*width);
    gsl_matrix_set_zero(fft_kernel);

    // when zero-padding, we must ensure that the center of the kernel
    // is copied on the corners of the padded image
    // There are four quadrants to copy

    // The columns on the left of the center are put on the extreme right of fft_kernel
    for(unsigned int i = 0 ; i < int(kernel->size2/2) ; ++i)
    {
        // The top rows of kernel are put on the bottom rows of fft_kernel
        gsl_vector_memcpy(&gsl_matrix_subcolumn(fft_kernel,fft_kernel->size2-2*int(kernel->size2/2)+2*i,fft_kernel->size1 - int(kernel->size1/2),int(kernel->size1/2)).vector,&gsl_matrix_subcolumn(kernel, i, 0, int(kernel->size1/2)).vector);
        // The bottom rows of the kernel are put on the top rows of fft_kernel
        gsl_vector_memcpy(&gsl_matrix_subcolumn(fft_kernel,fft_kernel->size2-2*int(kernel->size2/2)+2*i,0,int((kernel->size1+1)/2)).vector,&gsl_matrix_subcolumn(kernel, i, int(kernel->size1/2), int((kernel->size1+1)/2) ).vector);
    }
    // The columns on the right of the center are put on the extreme left of fft_kernel
    for(unsigned int i = int(kernel->size2/2) ; i < kernel->size2 ; ++i)
    {
        // The top rows of kernel are put on the bottom rows of fft_kernel
        gsl_vector_memcpy(&gsl_matrix_subcolumn(fft_kernel,2*(i-int(kernel->size2/2)),fft_kernel->size1 - int(kernel->size1/2),int(kernel->size1/2)).vector,&gsl_matrix_subcolumn(kernel, i, 0, int(kernel->size1/2)).vector);
        // The bottom rows of the kernel are put on the top rows of fft_kernel
        gsl_vector_memcpy(&gsl_matrix_subcolumn(fft_kernel,2*(i-int(kernel->size2/2)),0,int((kernel->size1+1)/2)).vector,&gsl_matrix_subcolumn(kernel, i, int(kernel->size1/2), int((kernel->size1+1)/2)).vector);
    }

    // We need to compute a 2D fft, which is performed by performing several 1D FFT
    // 1 - Apply the FFT on the lines
    // 2 - Apply the FFT on the columns of the previous result
    for(int i = 0 ; i < height ; ++i)
    {
        // Apply the FFT on the line i
        gsl_fft_complex_forward (&fft_src->data[i*2*width],1, width, wv_line, ws_line);
        gsl_fft_complex_forward (&fft_kernel->data[i*2*width],1, width, wv_line, ws_line);
    }
    for(int j = 0 ; j < width ; ++j)
    {
        // Apply the FFT on the column j
        gsl_fft_complex_forward (&fft_src->data[2*j],width, height, wv_column, ws_column);
        gsl_fft_complex_forward (&fft_kernel->data[2*j],width, height, wv_column, ws_column);
    }

    // Their element-wise product : be carefull, the matrices hold complex numbers !
    double rea, ima, reb, imb;
    for(int i = 0 ; i < height ; ++i)
    {
        for(int j = 0 ; j < width ; ++j)
        {
            rea = gsl_matrix_get(fft_src, i, 2*j);
            ima = gsl_matrix_get(fft_src, i, 2*j+1);
            reb = gsl_matrix_get(fft_kernel, i, 2*j);
            imb = gsl_matrix_get(fft_kernel, i, 2*j+1);
            gsl_matrix_set(fft_src, i, 2*j, rea * reb - ima * imb);
            gsl_matrix_set(fft_src, i, 2*j+1, rea * imb + ima * reb);
        }
    }

    // And the inverse FFT, which is done in the similar way as before
    for(int i = 0 ; i < height ; ++i)
    {
        // Apply the FFT^{-1} on the line i
        gsl_fft_complex_inverse(&fft_src->data[i*2*width],1, width, wv_line, ws_line);
    }

    for(int j = 0 ; j < width ; ++j)
    {
        // Apply the FFT^{-1} on the column j
        gsl_fft_complex_inverse(&fft_src->data[2*j],width, height, wv_column, ws_column);
    }

    // And copy only the real part of fft_src in dst
    for(unsigned int i = 0 ; i < src->size2 ; ++i)
    {
        gsl_matrix_set_col(dst, i, &gsl_matrix_subcolumn(fft_src, 2*i,0,src->size1).vector);
    }

    gsl_fft_complex_workspace_free(ws_column);
    gsl_fft_complex_workspace_free(ws_line);
    gsl_fft_complex_wavetable_free(wv_column);
    gsl_fft_complex_wavetable_free(wv_line);

    gsl_matrix_free(fft_src);
    gsl_matrix_free(fft_kernel);
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
    int low_k, high_k, low_l, high_l;

    // For each pixel in the dest image
    for (i = 0 ; i < h_src ; i++)
    {
        low_k = max(0, i + h_filt/2 - h_src + 1);
        high_k = min(h_filt, i + h_filt/2 + 1);

        for (j = 0 ; j  < w_src ; j++)
        {
            low_l = max(0, j + w_filt/2 - w_src + 1);
            high_l = min(w_filt, j + w_filt/2 + 1);
            temp = 0.0;
            // We browse the kernel
            for (k = low_k ; k < high_k ; k++)
            {
                for(l = low_l ; l < high_l ; l++)
                {
                    temp += gsl_matrix_get (src, i + h_filt/2 - k, j + w_filt/2 - l)*gsl_matrix_get (kernel, k, l);
                }
            }
            gsl_matrix_set (dst, i, j, temp);
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

    gsl_matrix * kernel = setup_filter("Gaussian");
    gsl_matrix * dst = gsl_matrix_alloc(h_src, w_src);
    gsl_matrix * dst_optimal = gsl_matrix_alloc(h_src, w_src);
    gsl_matrix * dst_std = gsl_matrix_alloc(h_src, w_src);

    // And compute the linear convolution
    gettimeofday(&before, NULL);
    for(int i = 0 ; i < NB_REPETITIONS ; ++i)
        linear_convolution_fft_gsl(src, kernel, dst);
    gettimeofday(&after, NULL);
    sbefore = before.tv_sec + before.tv_usec * 1E-6;
    safter =after.tv_sec + after.tv_usec * 1E-6;
    total = safter - sbefore;
    std::cout <<"Linear convolution GSL - Total elapsed time: "<<total/NB_REPETITIONS <<" s \n";

    // And compute the linear convolution with an optimal size
    gettimeofday(&before, NULL);
    for(int i = 0 ; i < NB_REPETITIONS ; ++i)
        linear_convolution_fft_gsl_optimal(src, kernel, dst_optimal);
    gettimeofday(&after, NULL);
    sbefore = before.tv_sec + before.tv_usec * 1E-6;
    safter =after.tv_sec + after.tv_usec * 1E-6;
    total = safter - sbefore;
    std::cout <<"Linear convolution GSL, optimal size - Total elapsed time: "<<total/NB_REPETITIONS <<" s \n";

    save_image(src,"src.jpg");
    save_image(kernel, "kernel.jpg");
    save_image(dst_optimal,"result.jpg");

    // Perform a standard 2D linear convolution
    std_convolution(src, kernel, dst_std);

    // Numerical comparison
    double error_value;
    gsl_vector * diff = gsl_vector_alloc(h_src * w_src);

    gsl_vector_memcpy(diff, &gsl_vector_view_array(dst_std->data, h_src*w_src).vector);
    gsl_vector_sub(diff, &gsl_vector_view_array(dst->data, h_src*w_src).vector);
    error_value = sqrt(gsl_pow_2(gsl_blas_dnrm2(diff))/(double(diff->size)));
    std::cout << " RMS (std, gsl) : " << std::scientific << error_value;

    gsl_vector_memcpy(diff, &gsl_vector_view_array(dst_std->data, h_src*w_src).vector);
    gsl_vector_sub(diff, &gsl_vector_view_array(dst_optimal->data, h_src*w_src).vector);
    error_value = sqrt(gsl_pow_2(gsl_blas_dnrm2(diff))/(double(diff->size)));
    std::cout << " RMS (std, gsl optimal) : " << std::scientific << error_value;

    // Free the memory
    gsl_matrix_free(src);
    gsl_matrix_free(kernel);
    gsl_matrix_free(dst);
    gsl_matrix_free(dst_optimal);
    gsl_matrix_free(dst_std);
}
