// Compile with g++ -o main main.cc `pkg-config --libs --cflags fftw3 gsl` -lX11 -lpthread -O3 -Wall
// Convolve an image with a gaussian filter or a dog filter (see around l. 960) using a standard convolution, a SVD decomposition, a fft convolution (gsl and fftw)
// Example : ./main Images/lena.jpg

// TODO : 
// - Measure performances with or without the optimization on the size
// - Try with asymetric filters like sobel, gabor, etc..

// In case you experience issues, comment the following line to enable the assert()
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

//#include <complex.h>
#include <fftw3.h>

#include <string>

// To load an image
#include "CImg.h"

// To measure the execution time
#include <sys/time.h>

using namespace cimg_library;

#define W_FILT 64
#define H_FILT 64
#define cmp_threshold 1e-10
#define NB_REPETITIONS 30

inline int min(int a, int b) { return a < b ? a : b;}
inline int max(int a, int b) { return a > b ? a : b;}

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

typedef enum
{
    ZERO_PADDING,
    WRAP_PADDING
} PaddingMode;


/* ******************************************************************************************************** */
/*          Tool functions to find optimal sizes for the images to convolve with fft and zero-padding       */
/* ******************************************************************************************************** */

int GSL_FACTORS[7] = {7,6 ,5,4,3,2,0}; // end with zero to detect the end of the array
int FFTW_FACTORS[7] = {13,11,7,5,3,2,0};

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
    bool is_optimal = true;
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


/* ******************************************************************************************************** */
/*          SVD :    DOES NOT WORK PROPERLY                                                                 */
/*  - svd_convolution : performs a convolution using the SVD decomposition of the kernel                    */
/* ******************************************************************************************************** */

void convolve_1d(const gsl_matrix * src,double eigen_value,const gsl_vector * conv_horiz,const gsl_vector * conv_vert,gsl_matrix * dst)
{
    // Convolution with zero padding
    int i,j,k,l;
    int h_src = src->size1;
    int w_src = src->size2;
    int h_filt = conv_horiz->size;
    int w_filt = conv_vert->size;
    
    gsl_matrix * dst_1D_tmp = gsl_matrix_alloc(h_src,w_src);
    
    int low_l, high_l;
    int low_k, high_k;
    double temp;
    
    for (i = 0 ; i < h_src ; i++)
    {
        low_l = max(0, h_filt/2.0 - i);
        high_l = min(h_filt, h_src + h_filt/2.0 - i);

        for (j = 0 ; j  < w_src ; j++)
	{ 
            temp = 0.0;

            // We browse the kernel
            for (l = low_l ; l < high_l ; l++)
                temp += gsl_matrix_get (src, int(i+ l - h_filt/2.0),j)*gsl_vector_get (conv_horiz, l);
            gsl_matrix_set (dst_1D_tmp, i, j, eigen_value*temp);
	}
    }
    

    // second with conv_vert
    for (j = 0 ; j  < w_src ; j++)
    {
        low_k = max(0, w_filt/2.0 - j);
        high_k = min(w_filt, w_src + w_filt/2.0 - j);
        for (i = 0 ; i < h_src ; i++)
	{
            temp = 0.0;
            // We browse the kernel
            for (k = low_k ; k < high_k ; k++)
                temp += gsl_matrix_get (dst_1D_tmp, i, int(j+ k - w_filt/2.0))*gsl_vector_get (conv_vert, k);
            gsl_matrix_set (dst, i, j, temp);
	}
    }
    gsl_matrix_free(dst_1D_tmp);

}

void svd_convolution(gsl_matrix * src, gsl_matrix * kernel, gsl_matrix * dst, PaddingMode pm = ZERO_PADDING)
{
    if(pm == ZERO_PADDING)
    {
        int i,j,k,l,m;
        int h_src = src->size1;
        int w_src = src->size2;
        int h_filt = kernel->size1;
        int w_filt = kernel->size2;
        double eigen_value;
        int low_l, high_l;
        int low_k, high_k;
        double temp;

        gsl_matrix * U = gsl_matrix_alloc(h_filt,w_filt);
        gsl_matrix * V = gsl_matrix_alloc(w_filt,w_filt);
        gsl_matrix * dst_1D_tmp = gsl_matrix_alloc(h_src,w_src);
        gsl_vector * S = gsl_vector_alloc(w_filt);
        gsl_vector * work = gsl_vector_alloc(w_filt);
        gsl_vector * conv_horiz = gsl_vector_alloc(h_filt);
        gsl_vector * conv_vert = gsl_vector_alloc(w_filt);

        gsl_matrix * dst_tmp = gsl_matrix_alloc(h_src,w_src);

        gsl_matrix_set_zero(dst);

        // We must copy the kernel since gsl_linalg_SV_decomp
        // erase the provided kernel
        gsl_matrix_memcpy(U,kernel);

        // Decompose the matrix
        gsl_linalg_SV_decomp (U,V,S,work);

        // Compute the number of non-null values of S
        int rank = 0;

        for(i = 0 ; i < w_filt ; i++)
            if( gsl_vector_get(S,i) > cmp_threshold)
                rank ++;

        // For each strictly postivie eigen value, we perform the convolutions
        // and add the result to the final result
        for(m = 0 ; m < rank ; m ++)
        {
            gsl_matrix_get_col(conv_horiz,U,m);
            gsl_matrix_get_col(conv_vert,V,m);
            eigen_value = gsl_vector_get(S,m);

            for (i = 0 ; i < h_src ; i++)
            {
                low_l = max(0, h_filt/2 - i);
                high_l = min(h_filt, h_src + h_filt/2 - i);

                for (j = 0 ; j  < w_src ; j++)
                {
                    temp = 0.0;
                    // We browse the kernel
                    for (l = low_l ; l < high_l ; l++)
                        temp += gsl_matrix_get (src, int(i+ l - h_filt/2),j)*gsl_vector_get (conv_horiz, l);
                    gsl_matrix_set (dst_1D_tmp, i, j, eigen_value*temp);
                }
            }

            // second with conv_vert
            for (j = 0 ; j  < w_src ; j++)
            {
                low_k = max(0, w_filt/2 - j);
                high_k = min(w_filt, w_src + w_filt/2 - j);
                for (i = 0 ; i < h_src ; i++)
                {
                    temp = 0.0;
                    // We browse the kernel
                    for (k = low_k ; k < high_k ; k++)
                        temp += gsl_matrix_get (dst_1D_tmp, i, int(j+ k - w_filt/2))*gsl_vector_get (conv_vert, k);
                    gsl_matrix_set (dst, i, j, gsl_matrix_get(dst,i ,j) + temp);
                }
            }
        }

        gsl_matrix_free (U);
        gsl_matrix_free (V);
        gsl_matrix_free(dst_1D_tmp);
        gsl_matrix_free (dst_tmp);
        gsl_vector_free (S);
        gsl_vector_free (work);
        gsl_vector_free (conv_horiz);
        gsl_vector_free (conv_vert);
    }
    else if( pm == WRAP_PADDING)
    {
        int i,j,k,l,m;
        int h_src = src->size1;
        int w_src = src->size2;
        int h_src_padded = src->size1 + kernel->size1;
        int w_src_padded = src->size2 + kernel->size2;
        int h_filt = kernel->size1;
        int w_filt = kernel->size2;
        int rank;
        double temp;
        double eigen_value;

        gsl_matrix * dst_1D_tmp = gsl_matrix_alloc(h_src_padded,w_src_padded);
        gsl_matrix * U = gsl_matrix_alloc(h_filt,w_filt);
        gsl_matrix * V = gsl_matrix_alloc(w_filt,w_filt);
        gsl_vector * S = gsl_vector_alloc(w_filt);
        gsl_vector * work = gsl_vector_alloc(w_filt);
        gsl_vector * conv_horiz = gsl_vector_alloc(h_filt);
        gsl_vector * conv_vert = gsl_vector_alloc(w_filt);

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

        //save_image(src_padded, "src_padded.jpg");
        gsl_matrix_set_zero(dst);

        // We must copy the kernel since gsl_linalg_SV_decomp
        // erase the provided kernel
        gsl_matrix_memcpy(U,kernel);

        // Decompose the matrix
        gsl_linalg_SV_decomp (U,V,S,work);

        // Compute the number of non-null values of S
        rank = 0;
        for(i = 0 ; i < w_filt ; i++)
            if( gsl_vector_get(S,i) > cmp_threshold)
                rank ++;

        for(m = 0 ; m < rank ; m ++)
        {
            eigen_value = gsl_vector_get(S, m);
            gsl_matrix_get_col(conv_horiz,U,m);
            gsl_matrix_get_col(conv_vert,V,m);

            // The first pass is done only on the lines
            // of the padded image between h_filt/2 and h_src_padded - h_filt/2
            // The other lines will not be used for the second pass
            for (i = h_filt/2 ; i < h_src_padded-h_filt/2 ; i++)
            {
                for (j = 0 ; j  < w_src_padded ; j++)
                {
                    temp = 0.0;
                    // We browse the kernel
                    for (l = 0 ; l < h_filt ; l++)
                        temp += gsl_matrix_get (src_padded, i + l - h_filt/2,j)*gsl_vector_get (conv_horiz, l);
                    gsl_matrix_set (dst_1D_tmp, i, j, eigen_value*temp);
                }
            }

            // second with conv_vert only on the part corresponding to the original image
            for (j = w_filt/2 ; j < w_src_padded - w_filt/2 ; j++)
            {
                //high_k = min(w_filt, w_src_padded + w_filt/2.0 - j);
                for (i = h_filt/2 ; i < h_src_padded - h_filt/2 ; i++)
                {
                    temp = 0.0;
                    // We browse the kernel
                    for (k = 0 ; k < w_filt ; k++)
                        temp += gsl_matrix_get (dst_1D_tmp, i, j + k - w_filt/2)*gsl_vector_get (conv_vert, k);
                    // printf("%i %i ; %i , %i \n", i, j, i-h_filt/2, j-w_filt/2);
                    gsl_matrix_set (dst, i-h_filt/2, j-w_filt/2, gsl_matrix_get(dst, i-h_filt/2, j-w_filt/2) + temp);
                }
            }
        }

        gsl_matrix_free(src_padded);
        gsl_matrix_free(dst_1D_tmp);
        gsl_matrix_free(U);
        gsl_matrix_free(V);
        gsl_vector_free(S);
        gsl_vector_free(work);
        gsl_vector_free(conv_horiz);
        gsl_vector_free(conv_vert);
    }
    else
    {
        printf("svd_convolution : unrecognized padding mode %i \n", pm);
    }
}

// Useless for now : an attempt to check the performances between the complex fourier transform, and the half-complex transforms
// using real data (i.e. not complex) as an input

//void fft_convolution_real(const gsl_matrix * src, const gsl_matrix * kernel, gsl_matrix * dst)
//{
//    assert(kernel->size1 < src->size1 && kernel->size2 < src->size2 && dst->size1 == src->size1 && dst->size2 == dst->size2);
//
//    int height = src->size1;
//    int width = src->size2;
//
//    // Create some required matrices
//    gsl_fft_real_workspace * ws_column = gsl_fft_real_workspace_alloc(height);
//    gsl_fft_real_workspace * ws_line = gsl_fft_real_workspace_alloc(width);
//    gsl_fft_real_wavetable * wv_column = gsl_fft_real_wavetable_alloc(height);
//    gsl_fft_real_wavetable * wv_line = gsl_fft_real_wavetable_alloc(width);
//
//    gsl_fft_halfcomplex_wavetable * wv_hc_column = gsl_fft_halfcomplex_wavetable_alloc(height);
//    gsl_fft_halfcomplex_wavetable * wv_hc_line = gsl_fft_halfcomplex_wavetable_alloc(width);
//
//    gsl_matrix * fft_src = gsl_matrix_alloc(height, width);
//    gsl_matrix_memcpy(fft_src, src);
//    // We padd the kernel so that it has the same size as the src image
//    // we consider that the kernel is smalled than src
//    gsl_matrix * fft_kernel = gsl_matrix_alloc(height, width);
//
//    gsl_matrix_set_zero(fft_kernel);
//    gsl_matrix_memcpy(&gsl_matrix_submatrix(fft_kernel,(height - kernel->size1)/2,(width - kernel->size2)/2, kernel->size1,kernel->size2).matrix, kernel);
//
//    // We need to compute a 2D fft, which is performed by performing several 1D FFT
//    // 1 - Apply the FFT on the lines
//    // 2 - Apply the FFT on the columns of the previous result
//    int i, j;
//    for(i = 0 ; i < height ; ++i)
//    {
//        // Apply the FFT on the line i
//        gsl_fft_real_transform (&fft_src->data[i*width],1, width, wv_line, ws_line);
//        gsl_fft_real_transform (&fft_kernel->data[i*width],1, width, wv_line, ws_line);
//    }
//    for(j = 0 ; j < width ; ++j)
//    {
//        // Apply the FFT on the column j
//        gsl_fft_real_transform (&fft_src->data[j],width, height, wv_column, ws_column);
//        gsl_fft_real_transform (&fft_kernel->data[j],width, height, wv_column, ws_column);
//    }
//    // The product of the two 2D-FT must take care of the half-complex representation provided by gsl_fft_real_transform
//    // obviously does not work : gsl_matrix_mul_elements(fft_src, fft_kernel);
//    // HOW TO PERFORM THE MULTIPLICATION ON THE HALF-COMPLEX without unpacking them ???
//
//    //gsl_fft_halfcomplex_unpack(fft_src->data,
//
//
//    // And the inverse FFT, which is done in the similar way as before
//    for(i = 0 ; i < height ; ++i)
//    {
//        // Apply the FFT^{-1} on the line i
//        gsl_fft_halfcomplex_backward(&fft_src->data[i*width],1, width, wv_hc_line, ws_line);
//    }
//
//    for(j = 0 ; j < width ; ++j)
//    {
//        // Apply the FFT^{-1} on the column j
//        gsl_fft_halfcomplex_backward(&fft_src->data[j],width, height, wv_hc_column, ws_column);
//    }
//
//    gsl_matrix_memcpy(dst, fft_src);
//
//    gsl_fft_real_workspace_free(ws_column);
//    gsl_fft_real_workspace_free(ws_line);
//    gsl_fft_real_wavetable_free(wv_column);
//    gsl_fft_real_wavetable_free(wv_line);
//
//    gsl_fft_halfcomplex_wavetable_free(wv_hc_column);
//    gsl_fft_halfcomplex_wavetable_free(wv_hc_line);
//
//    gsl_matrix_free(fft_src);
//    gsl_matrix_free(fft_kernel);
//}

/* ******************************************************************************************************** */
/*          FFT - GSL :                                                                                     */
/*  - test_fft_cplx : it performs the forward and backward transform to check if this leads to identity     */
/*  - fft_convolution_cplx : performs a convolution with Zero or Wrap padding                               */
/* ******************************************************************************************************** */

void test_fft_cplx(gsl_matrix * src)
{
    int height = src->size1;
    int width = src->size2;

    save_image(src, "test_fft_src.jpg");
    // Create some required matrices
    gsl_fft_complex_workspace * ws_column = gsl_fft_complex_workspace_alloc(height);
    gsl_fft_complex_workspace * ws_line = gsl_fft_complex_workspace_alloc(width);
    gsl_fft_complex_wavetable * wv_column = gsl_fft_complex_wavetable_alloc(height);
    gsl_fft_complex_wavetable * wv_line = gsl_fft_complex_wavetable_alloc(width);

    // We need a real and imaginary part
    gsl_matrix * fft_src = gsl_matrix_alloc(height, 2*width);
    // Copy the real part and let the imaginary part to 0
    gsl_vector_memcpy(&gsl_vector_view_array_with_stride(fft_src->data,2,height * width).vector, &gsl_vector_view_array(src->data,width*height).vector);

    gsl_matrix * res = gsl_matrix_alloc(height ,width);

    int i, j;
    for(i = 0 ; i < height ; ++i)
    {
        // Apply the FFT on the line i
        gsl_fft_complex_forward (&fft_src->data[i*2*width],1 , width, wv_line, ws_line);
    }

    for(j = 0 ; j < width ; ++j)
    {
        // Apply the FFT on the column j
        gsl_fft_complex_forward (&fft_src->data[2*j],width, height, wv_column, ws_column);
    }

    // And the inverse FFT, which is done in the similar way as before
    for(i = 0 ; i < height ; ++i)
    {
        // Apply the FFT^{-1} on the line i
        gsl_fft_complex_inverse(&fft_src->data[i*2*width],1, width, wv_line, ws_line);
    }

    for(j = 0 ; j < width ; ++j)
    {
        // Apply the FFT^{-1} on the column j
        gsl_fft_complex_inverse(&fft_src->data[2*j],width, height, wv_column, ws_column);
    }

    for(i = 0 ; i < width ; ++i)
    {
        gsl_matrix_set_col(res, i, &gsl_matrix_column(fft_src, 2*i).vector);
    }

    save_image(res, "test_fft_ifft.jpg");

    gsl_fft_complex_workspace_free(ws_column);
    gsl_fft_complex_workspace_free(ws_line);
    gsl_fft_complex_wavetable_free(wv_column);
    gsl_fft_complex_wavetable_free(wv_line);
    gsl_matrix_free(fft_src);
    gsl_matrix_free(res);
}

void fft_convolution_cplx(gsl_matrix * src, gsl_matrix * kernel, gsl_matrix * dst, PaddingMode pm = ZERO_PADDING)
{
    assert(kernel->size1 <= src->size1 && kernel->size2 <= src->size2 && dst->size1 == src->size1 && dst->size2 == dst->size2);

    int height;
    int width;

    switch(pm)
    {
    case WRAP_PADDING:
        height = src->size1;
        width = src->size2;
        break;
    case ZERO_PADDING:
        // We assume the kernel matrix is defined such that the filter is 0 on the borders
        // To avoid the image is polluted on the borders because of the cycling considered by the FFT, we pad some zeros on one end of the image
        // of at least the half size of the filter
        /*height = src->size1 + int(kernel->size1+1)/2;
        width = src->size2 + int(kernel->size2+1)/2;*/
        // We may even modify the size to make it compliant with the optimized FFT size factors : 2 , 3 , 4 , 5 , 6 , 7
        //height = find_closest_factor(src->size1 + int(kernel->size1+1)/2, GSL_FACTORS);
        //width = find_closest_factor(src->size2 + int(kernel->size2+1)/2, GSL_FACTORS);
        height = find_closest_factor(src->size1 + kernel->size1, GSL_FACTORS);
        width = find_closest_factor(src->size2 + kernel->size2, GSL_FACTORS);
        break;
    default:
        printf("fft_cplx : The padding mode %i is not implemented \n", pm);
        return;
    }

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
    // we consider that the kernel is smalled than src
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

/* ******************************************************************************************************** */
/*          FFTW3 :                                                                                         */
/*  - test_fftw3 : it performs the forward and backward transform to check if this leads to identity        */
/*  - fftw3_convolution_cplx :performs a convolution with Zero or Wrap padding                              */
/* ******************************************************************************************************** */
void test_fftw3(gsl_matrix * src)
{
    int height;
    int width;

    height = src->size1;
    width = src->size2;

    // Create the required objects for FFTW
    fftw_complex *in_src, *out_src;
    fftw_plan p_forw;
    fftw_plan p_back;

    in_src = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * height * width);
    out_src = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * height * width);
    p_forw = fftw_plan_dft_2d(height, width, in_src, out_src, FFTW_FORWARD, FFTW_ESTIMATE);
    p_back = fftw_plan_dft_2d(height, width, out_src, in_src, FFTW_BACKWARD, FFTW_ESTIMATE);

    for(int i = 0 ; i < height ; ++i)
    {
        for(int j = 0 ; j < width ; ++j)
        {
            in_src[i * width + j][0] = gsl_matrix_get(src, i, j);
            in_src[i * width + j][1] = 0.0;
        }
    }

    // Compute the forward fft
    fftw_execute(p_forw);

    // Compute the backward fft
    fftw_execute(p_back);

    // Now we just need to copy the right part of in_src into dst
    gsl_matrix * dst =gsl_matrix_alloc(height, width);
    for(int i  = 0 ; i < height ; ++i)
    {
        for(int j = 0 ; j < width ; ++j)
        {
            gsl_matrix_set(dst, i, j, in_src[i * width + j][0]/ double(width * height));
        }
    }
    save_image(dst, "test_fftw3_dst.png");
    gsl_matrix_free(dst);

    fftw_destroy_plan(p_forw);
    fftw_destroy_plan(p_back);
    fftw_free(in_src);
    fftw_free(out_src);
}

void fftw3_convolution_cplx(gsl_matrix * src, gsl_matrix * kernel, gsl_matrix * dst, PaddingMode pm = ZERO_PADDING)
{  
    assert(kernel->size1 <= src->size1 && kernel->size2 <= src->size2 && dst->size1 == src->size1 && dst->size2 == dst->size2);

    int height;
    int width;

    switch(pm)
    {
    case WRAP_PADDING:
        height = src->size1;
        width = src->size2;
        break;
    case ZERO_PADDING:
        // We assume the kernel matrix is defined such that the filter is 0 on the borders
        // To avoid the image is polluted on the borders because of the cycling considered by the FFT, we pad some zeros on one end of the image
        // of at least the half size of the filter
        /*height = src->size1 + int((kernel->size1+1)/2);
        width = src->size2 + int((kernel->size2+1)/2);*/
        // We may pad it with additional zeros to get a size compliant with the optimized size of FFTW 2, 3, 5, and 7
        height = find_closest_factor(src->size1 + int((kernel->size1+1)/2), FFTW_FACTORS);
        width = find_closest_factor(src->size2 + int((kernel->size2+1)/2), FFTW_FACTORS);
        break;
    default:
        printf("fft_cplx : The padding mode %i is not implemented \n", pm);
        return;
    }

    // Create the required objects for FFTW
    fftw_complex *in_src, *out_src, *in_kernel, *out_kernel;
    fftw_plan p_forw;
    fftw_plan p_kernel_forw;
    fftw_plan p_back;

    in_src = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * height * width);
    out_src = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * height * width);
    in_kernel = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * height * width);
    out_kernel = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * height * width);
    p_forw = fftw_plan_dft_2d(height, width, in_src, out_src, FFTW_FORWARD, FFTW_ESTIMATE);
    p_kernel_forw = fftw_plan_dft_2d(height, width, in_kernel, out_kernel, FFTW_FORWARD, FFTW_ESTIMATE);
    p_back = fftw_plan_dft_2d(height, width, out_src, in_src, FFTW_BACKWARD, FFTW_ESTIMATE);

    // We need to fill the real part of in_src with the src image
    for(int i = 0 ; i < height ; ++i)
    {
        for(int j = 0 ; j < width ; ++j)
        {
            if( i < src->size1 && j < src->size2)
                in_src[i * width + j][0] = gsl_matrix_get(src, i, j);
            else
                in_src[i * width + j][0] = 0.0;
            in_src[i * width + j][1] = 0.0;
        }
    }

    // We padd the kernel so that it has the same size as the src image
    for(int i = 0 ; i < height ; ++i)
    {
        for(int j = 0 ; j < width ; ++j)
        {
            if(i < int((kernel->size1+1)/2))
            {
                if(j < int((kernel->size2+1)/2))
                {
                    in_kernel[i * width + j][0] = gsl_matrix_get(kernel, i + int(kernel->size1/2), j+ int(kernel->size2/2) );
                    in_kernel[i * width + j][1] = 0.0;
                }
                else if( j < width - int(kernel->size2/2))
                {
                    in_kernel[i * width + j][0] = 0.0;
                    in_kernel[i * width + j][1] = 0.0;
                }
                else
                {
                    in_kernel[i * width + j][0] = gsl_matrix_get(kernel, i + kernel->size1/2, (j - width + int(kernel->size2/2)));
                    in_kernel[i * width + j][1] = 0.0;
                }
            }
            else if( i < height - int(kernel->size1/2) )
            {
                in_kernel[i * width + j][0] = 0.0;
                in_kernel[i * width + j][1] = 0.0;
            }
            else
            {
                if(j < int((kernel->size2+1)/2) )
                {
                    in_kernel[i * width + j][0] = gsl_matrix_get(kernel, i - (height -int(kernel->size1/2)), j+ int(kernel->size2/2) );
                    in_kernel[i * width + j][1] = 0.0;
                }
                else if( j < width - int(kernel->size2/2))
                {
                    in_kernel[i * width + j][0] = 0.0;
                    in_kernel[i * width + j][1] = 0.0;
                }
                else
                {
                    in_kernel[i * width + j][0] = gsl_matrix_get(kernel, i - (height -int(kernel->size1/2)), (j - width + int(kernel->size2/2)));
                    in_kernel[i * width + j][1] = 0.0;
                }
            }
        }
    }

    // Compute the forward fft
    fftw_execute(p_forw);
    fftw_execute(p_kernel_forw);

    double rea, ima, reb, imb;
    // Compute the element-wise product
    for(int i = 0 ; i < width * height ; ++i)
    {
        rea = out_src[i][0];
        ima = out_src[i][1];
        reb = out_kernel[i][0];
        imb = out_kernel[i][1];
        out_src[i][0] = rea * reb - ima * imb;
        out_src[i][1] = rea * imb + ima * reb;
    }

    // Compute the backward fft
    fftw_execute(p_back);

    // Now we just need to copy the right part of in_src into dst
    for(int i  = 0 ; i < dst->size1 ; ++i)
    {
        for(int j = 0 ; j < dst->size2 ; ++j)
        {
            gsl_matrix_set(dst, i, j, in_src[i * width + j][0]/ double(width * height));
        }
    }

    fftw_destroy_plan(p_forw);
    fftw_destroy_plan(p_kernel_forw);
    fftw_destroy_plan(p_back);
    fftw_free(in_src); fftw_free(out_src);
    fftw_free(in_kernel); fftw_free(out_kernel);
}

/* ******************************************************************************************************** */
/*                   Standard convolution, with Zero or Wrap padding                                        */
/* ******************************************************************************************************** */

void std_convolution(gsl_matrix * src, gsl_matrix * kernel, gsl_matrix * dst, PaddingMode pm = ZERO_PADDING)
{
    if(pm == ZERO_PADDING)
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
    else if(pm == WRAP_PADDING)
    {
        int i,j,k,l;
        int h_src = src->size1;
        int w_src = src->size2;
        int h_filt = kernel->size1;
        int w_filt = kernel->size2;
        double temp;

        int i_src, j_src;
        for (i = 0 ; i < src->size1 ; ++i)
        {
            for (j = 0 ; j  < src->size2 ; ++j)
            {
                temp = 0.0;
                // We browse the kernel
                for (k = 0 ; k < h_filt  ; ++k)
                {
                    for(l = 0 ; l < w_filt ; ++l)
                    {
                        if((i + h_filt/2 - k) < 0)
                            i_src = (i + h_filt/2 - k) + src->size1;
                        else if(i + h_filt/2 - k >= src->size1)
                            i_src = (i + h_filt/2 - k) - src->size1;
                        else
                            i_src = (i + h_filt/2 - k);
                        if((j + w_filt/2 - l) < 0)
                            j_src = (j + w_filt/2 - l) + src->size2;
                        else if((j + w_filt/2 - l) >= src->size2)
                            j_src = (j + w_filt/2 - l) - src->size2;
                        else
                            j_src = (j + w_filt/2 - l);
                        temp += gsl_matrix_get (src, i_src, j_src)*gsl_matrix_get (kernel, k , l);
                    }
                }
                gsl_matrix_set (dst, i, j, temp);
            }
        }
    }
    else
    {
        printf("Padding mode %i is not yet implemented \n", pm);
    }
}

/* ******************************************************************************************************** */
/*                   Main : it loads an image and convolves it with a gaussian filter                       */
/* ******************************************************************************************************** */

// Benchmark function
// it simply creates different random images and random kernels and convolve them
void benchmark(int src_size_min, int src_size_max, int kernel_size_min, int kernel_size_max)
{
    int i;
    struct timeval before,after;
    double sbefore, safter, total;

    int w_src, h_src;
    int w_filt, h_filt;

    printf("Width_src Height_src Width_kernel Height_kernel FFT_gsl_zero FFT_gsl_wrap FFTW3_zero FFTW_wrap STD_zero STD_wrap\n");
    for(w_src = src_size_min, h_src = src_size_min ; w_src < src_size_max ; w_src++, h_src++)
    {
        for(w_filt = kernel_size_min, h_filt = kernel_size_min ; w_filt < min(kernel_size_max, w_src+1) ; w_filt++, h_filt++)
        {
            gsl_matrix * src = gsl_matrix_alloc(h_src,w_src);
            gsl_matrix * kernel = gsl_matrix_alloc(h_filt,w_filt);
            gsl_matrix * dst_fft_zero = gsl_matrix_alloc(h_src,w_src);
            gsl_matrix * dst_fftw3_zero = gsl_matrix_alloc(h_src,w_src);
            gsl_matrix * dst_std_zero = gsl_matrix_alloc(h_src,w_src);
            gsl_matrix * dst_fft_wrap = gsl_matrix_alloc(h_src,w_src);
            gsl_matrix * dst_fftw3_wrap = gsl_matrix_alloc(h_src,w_src);
            gsl_matrix * dst_std_wrap = gsl_matrix_alloc(h_src,w_src);

            printf("%i %i %i %i ", w_src, h_src, w_filt, h_filt);

            gettimeofday(&before, NULL);
            for(i = 0 ; i < NB_REPETITIONS ; ++i)
                fft_convolution_cplx(src,kernel,dst_fft_zero,ZERO_PADDING);
            gettimeofday(&after, NULL);
            sbefore = before.tv_sec + before.tv_usec * 1E-6;
            safter =after.tv_sec + after.tv_usec * 1E-6;
            total = safter - sbefore;
            printf("%f ", total);

            gettimeofday(&before, NULL);
            for(i = 0 ; i < NB_REPETITIONS ; ++i)
                fft_convolution_cplx(src,kernel,dst_fft_wrap,WRAP_PADDING);
            gettimeofday(&after, NULL);
            sbefore = before.tv_sec + before.tv_usec * 1E-6;
            safter =after.tv_sec + after.tv_usec * 1E-6;
            total = safter - sbefore;
            printf("%f ", total);

            /************************************/
            /********** FFTW convolution *********/
            /************************************/
            gettimeofday(&before, NULL);
            for(i = 0 ; i < NB_REPETITIONS ; ++i)
                fftw3_convolution_cplx(src,kernel,dst_fftw3_zero,ZERO_PADDING);
            gettimeofday(&after, NULL);
            sbefore = before.tv_sec + before.tv_usec * 1E-6;
            safter =after.tv_sec + after.tv_usec * 1E-6;
            total = safter - sbefore;
            printf("%f ", total);

            gettimeofday(&before, NULL);
            for(i = 0 ; i < NB_REPETITIONS ; ++i)
                fftw3_convolution_cplx(src,kernel,dst_fftw3_wrap,WRAP_PADDING);
            gettimeofday(&after, NULL);
            sbefore = before.tv_sec + before.tv_usec * 1E-6;
            safter =after.tv_sec + after.tv_usec * 1E-6;
            total = safter - sbefore;
            printf("%f ", total);

            /************************************/
            /****** Standard convolution ********/
            /************************************/
           /* gettimeofday(&before, NULL);
            for(i = 0 ; i < NB_REPETITIONS ; ++i)
                std_convolution(src,kernel,dst_std_zero, ZERO_PADDING);
            gettimeofday(&after, NULL);
            sbefore = before.tv_sec + before.tv_usec * 1E-6;
            safter =after.tv_sec + after.tv_usec * 1E-6;
            total = safter - sbefore;*/
            printf("%f ", 0.0);

            /*gettimeofday(&before, NULL);
            for(i = 0 ; i < NB_REPETITIONS ; ++i)
                std_convolution(src,kernel,dst_std_wrap, WRAP_PADDING);
            gettimeofday(&after, NULL);
            sbefore = before.tv_sec + before.tv_usec * 1E-6;
            safter =after.tv_sec + after.tv_usec * 1E-6;
            total = safter - sbefore;*/

            printf("%f ", 0.0);

            printf("\n");
            gsl_matrix_free (src);
            gsl_matrix_free (dst_fft_zero);
            gsl_matrix_free (dst_fftw3_zero);
            gsl_matrix_free (dst_std_zero);
            gsl_matrix_free (dst_fft_wrap);
            gsl_matrix_free (dst_fftw3_wrap);
            gsl_matrix_free (dst_std_wrap);
            gsl_matrix_free (kernel);
        }

    }

}

int main(int argc, char * argv[])
{
    if(argc != 2)
    {
        printf("Usage : \n");
        printf("   ./main <filename> \n");
        return 0;
    }

    //benchmark(144,512,10,512);

    int i, j;
    struct timeval before,after;
    double sbefore, safter, total;
    double error_value;
    int w_src, h_src;
    int w_filt, h_filt;
    char * filename = argv[1];

    // Read an image file
    CImg<unsigned char> image(filename);
    w_src = image.width();
    h_src = image.height();

    w_filt = W_FILT;
    h_filt = H_FILT;

    gsl_matrix * src = gsl_matrix_alloc(h_src,w_src);
    gsl_matrix * kernel = gsl_matrix_alloc(h_filt,w_filt);
    gsl_matrix * dst_svd_zero = gsl_matrix_alloc(h_src,w_src);
    gsl_matrix * dst_fft_zero = gsl_matrix_alloc(h_src,w_src);
    gsl_matrix * dst_fftw3_zero = gsl_matrix_alloc(h_src,w_src);
    gsl_matrix * dst_std_zero = gsl_matrix_alloc(h_src,w_src);
    gsl_matrix * dst_svd_wrap = gsl_matrix_alloc(h_src,w_src);
    gsl_matrix * dst_fft_wrap = gsl_matrix_alloc(h_src,w_src);
    gsl_matrix * dst_fftw3_wrap = gsl_matrix_alloc(h_src,w_src);
    gsl_matrix * dst_std_wrap = gsl_matrix_alloc(h_src,w_src);

    // src is the image to convolve
    for (i = 0 ; i < h_src ; i++)
        for (j = 0 ; j  < w_src ; j++)
            gsl_matrix_set(src,i,j,(int)(image(j,i,0)));

    int center_h = int((h_filt)/2);
    int center_w = int((w_filt)/2);
    for (i = 0; i < h_filt; i++)
    {
        for (j = 0; j < w_filt; j++)
        {
            // A gaussian filter
            //gsl_matrix_set (kernel, i, j, exp(-gsl_pow_2(i-center_h)/(2.0 * gsl_pow_2(h_filt/15.0))-gsl_pow_2(j-center_w)/(2.0 * gsl_pow_2(w_filt/15.0))));
            //gsl_matrix_set (kernel, i, j, exp(-gsl_pow_2(i-center_h)/(2.0 * gsl_pow_2(h_filt/1.0))-gsl_pow_2(j-center_w)/(2.0 * gsl_pow_2(w_filt/1.0))));
            // A dog filter
            //gsl_matrix_set (kernel, i, j, 0.95*exp(-6.0*(i-h_filt/2.0)*(i-h_filt/2.0)-6.0*(j-w_filt/2.0)*(j-w_filt/2.0))-0.65*exp(-(i-h_filt/2.0)*(i-h_filt/2.0)-(j-w_filt/2.0)*(j-w_filt/2.0)));
            gsl_matrix_set(kernel, i, j, rand() / double(RAND_MAX));
        }
    }
    /*gsl_matrix_set(kernel, center_h - 1,center_w-1, -1.0);
    gsl_matrix_set(kernel, center_h - 1,center_w, -2.0);
    gsl_matrix_set(kernel, center_h - 1,center_w+1, -3.0);
    gsl_matrix_set(kernel, center_h + 1,center_w-1, 3.0);
    gsl_matrix_set(kernel, center_h + 1,center_w, 5.0);
    gsl_matrix_set(kernel, center_h + 1,center_w+1, 1.0);*/

    save_image(kernel, "kernel.ppm");
    //test_fftw3(src);

    /************************************/
    /********** SVD convolution *********/
    /************************************/
    /*gettimeofday(&before, NULL);
    for(i = 0 ; i < NB_REPETITIONS ; ++i)
        svd_convolution(src,kernel,dst_svd_zero, ZERO_PADDING);
    gettimeofday(&after, NULL);
    sbefore = before.tv_sec + before.tv_usec * 1E-6;
    safter =after.tv_sec + after.tv_usec * 1E-6;
    total = safter - sbefore;
    std::cout <<"SVD Convolution - Zero padding - Total elapsed time: "<<total/NB_REPETITIONS <<" s \n";
    save_image(dst_svd_zero, "zero_svd_conv.jpg");

    gettimeofday(&before, NULL);
    for(i = 0 ; i < NB_REPETITIONS ; ++i)
        svd_convolution(src,kernel,dst_svd_wrap, WRAP_PADDING);
    gettimeofday(&after, NULL);
    sbefore = before.tv_sec + before.tv_usec * 1E-6;
    safter =after.tv_sec + after.tv_usec * 1E-6;
    total = safter - sbefore;
    std::cout <<"SVD Convolution - Wrap padding - Total elapsed time: "<<total/NB_REPETITIONS <<" s \n";
    save_image(dst_svd_wrap, "wrap_svd_conv.jpg");*/

    /************************************/
    /********** FFT convolution *********/
    /************************************/
    gettimeofday(&before, NULL);
    for(i = 0 ; i < NB_REPETITIONS ; ++i)
        fft_convolution_cplx(src,kernel,dst_fft_zero,ZERO_PADDING);
    gettimeofday(&after, NULL);
    sbefore = before.tv_sec + before.tv_usec * 1E-6;
    safter =after.tv_sec + after.tv_usec * 1E-6;
    total = safter - sbefore;
    std::cout <<"FFT Convolution - Zero padding - Total elapsed time: "<<total/NB_REPETITIONS <<" s \n";
    save_image(dst_fft_zero, "zero_fft_conv.jpg");

    gettimeofday(&before, NULL);
    for(i = 0 ; i < NB_REPETITIONS ; ++i)
        fft_convolution_cplx(src,kernel,dst_fft_wrap,WRAP_PADDING);
    gettimeofday(&after, NULL);
    sbefore = before.tv_sec + before.tv_usec * 1E-6;
    safter =after.tv_sec + after.tv_usec * 1E-6;
    total = safter - sbefore;
    std::cout <<"FFT Convolution - Wrap padding - Total elapsed time: "<<total/NB_REPETITIONS <<" s \n";
    save_image(dst_fft_wrap, "wrap_fft_conv.jpg");

    /************************************/
    /********** FFTW convolution *********/
    /************************************/
    gettimeofday(&before, NULL);
    for(i = 0 ; i < NB_REPETITIONS ; ++i)
        fftw3_convolution_cplx(src,kernel,dst_fftw3_zero,ZERO_PADDING);
    gettimeofday(&after, NULL);
    sbefore = before.tv_sec + before.tv_usec * 1E-6;
    safter =after.tv_sec + after.tv_usec * 1E-6;
    total = safter - sbefore;
    std::cout <<"FFTW3 Convolution - Zero padding - Total elapsed time: "<<total/NB_REPETITIONS <<" s \n";
    save_image(dst_fftw3_zero, "zero_fftw_conv.jpg");

    gettimeofday(&before, NULL);
    for(i = 0 ; i < NB_REPETITIONS ; ++i)
        fftw3_convolution_cplx(src,kernel,dst_fftw3_wrap,WRAP_PADDING);
    gettimeofday(&after, NULL);
    sbefore = before.tv_sec + before.tv_usec * 1E-6;
    safter =after.tv_sec + after.tv_usec * 1E-6;
    total = safter - sbefore;
    std::cout <<"FFTW3 Convolution - Wrap padding - Total elapsed time: "<<total/NB_REPETITIONS <<" s \n";
    save_image(dst_fftw3_wrap, "wrap_fftw_conv.jpg");

    /************************************/
    /****** Standard convolution ********/
    /************************************/
    gettimeofday(&before, NULL);
    for(i = 0 ; i < NB_REPETITIONS ; ++i)
        std_convolution(src,kernel,dst_std_zero, ZERO_PADDING);
    gettimeofday(&after, NULL);
    sbefore = before.tv_sec + before.tv_usec * 1E-6;
    safter =after.tv_sec + after.tv_usec * 1E-6;
    total = safter - sbefore;
    std::cout <<"Std convolution - Zero padding - Total elapsed time: "<<total/NB_REPETITIONS <<" s \n";
    save_image(dst_std_zero, "zero_std_conv.jpg");

    gettimeofday(&before, NULL);
    for(i = 0 ; i < NB_REPETITIONS ; ++i)
        std_convolution(src,kernel,dst_std_wrap, WRAP_PADDING);
    gettimeofday(&after, NULL);
    sbefore = before.tv_sec + before.tv_usec * 1E-6;
    safter =after.tv_sec + after.tv_usec * 1E-6;
    total = safter - sbefore;
    std::cout <<"Std convolution - Wrap padding - Total elapsed time: "<<total/NB_REPETITIONS <<" s \n";
    save_image(dst_std_wrap, "wrap_std_conv.jpg");

    /************************************/
    /****** Numerical comparison ********/
    /************************************/

    gsl_vector * diff = gsl_vector_alloc(dst_std_zero->size1*dst_std_zero->size2);

    gsl_vector_memcpy(diff, &gsl_vector_view_array(dst_std_zero->data, dst_std_zero->size1*dst_std_zero->size2).vector);
    gsl_vector_sub(diff, &gsl_vector_view_array(dst_svd_zero->data, dst_svd_zero->size1*dst_svd_zero->size2).vector);
    error_value = sqrt(gsl_pow_2(gsl_blas_dnrm2(diff))/(double(diff->size)));
    std::cout << "RMS(SVD, standard) - Zero padding : " << error_value << std::endl;

    gsl_vector_memcpy(diff, &gsl_vector_view_array(dst_std_zero->data, dst_std_zero->size1*dst_std_zero->size2).vector);
    gsl_vector_sub(diff, &gsl_vector_view_array(dst_fft_zero->data, dst_fft_zero->size1*dst_fft_zero->size2).vector);
    error_value = sqrt(gsl_pow_2(gsl_blas_dnrm2(diff))/(double(diff->size)));
    std::cout << "RMS(FFT, standard) - Zero padding : " << error_value << std::endl;

    gsl_vector_memcpy(diff, &gsl_vector_view_array(dst_std_zero->data, dst_std_zero->size1*dst_std_zero->size2).vector);
    gsl_vector_sub(diff, &gsl_vector_view_array(dst_fftw3_zero->data, dst_fftw3_zero->size1*dst_fftw3_zero->size2).vector);
    error_value = sqrt(gsl_pow_2(gsl_blas_dnrm2(diff))/(double(diff->size)));
    std::cout << "RMS(FFTW3, standard) - Zero padding : " << error_value << std::endl;

    gsl_vector_memcpy(diff, &gsl_vector_view_array(dst_std_wrap->data, dst_std_wrap->size1*dst_std_wrap->size2).vector);
    gsl_vector_sub(diff, &gsl_vector_view_array(dst_svd_wrap->data, dst_svd_wrap->size1*dst_svd_wrap->size2).vector);
    error_value = sqrt(gsl_pow_2(gsl_blas_dnrm2(diff))/(double(diff->size)));
    std::cout << "RMS(SVD, standard) - Wrap padding : " << error_value << std::endl;

    gsl_vector_memcpy(diff, &gsl_vector_view_array(dst_std_wrap->data, dst_std_wrap->size1*dst_std_wrap->size2).vector);
    gsl_vector_sub(diff, &gsl_vector_view_array(dst_fft_wrap->data, dst_fft_wrap->size1*dst_fft_wrap->size2).vector);
    error_value = sqrt(gsl_pow_2(gsl_blas_dnrm2(diff))/(double(diff->size)));
    std::cout << "RMS(FFT, standard) - Wrap padding : " << error_value << std::endl;

    gsl_vector_memcpy(diff, &gsl_vector_view_array(dst_std_wrap->data, dst_std_wrap->size1*dst_std_wrap->size2).vector);
    gsl_vector_sub(diff, &gsl_vector_view_array(dst_fftw3_wrap->data, dst_fftw3_wrap->size1*dst_fftw3_wrap->size2).vector);
    error_value = sqrt(gsl_pow_2(gsl_blas_dnrm2(diff))/(double(diff->size)));
    std::cout << "RMS(FFTW3, standard) - Wrap padding : " << error_value << std::endl;

    gsl_matrix_free (src);
    gsl_matrix_free (dst_svd_zero);
    gsl_matrix_free (dst_fft_zero);
    gsl_matrix_free (dst_fftw3_zero);
    gsl_matrix_free (dst_std_zero);
    gsl_matrix_free (dst_svd_wrap);
    gsl_matrix_free (dst_fft_wrap);
    gsl_matrix_free (dst_fftw3_wrap);
    gsl_matrix_free (dst_std_wrap);
    gsl_matrix_free (kernel);

    return 0;


}

