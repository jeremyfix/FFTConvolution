// Compile with g++ -o linear_convolution_fftw linear_convolution_fftw.cc `pkg-config --libs --cflags fftw3` -lX11 -lpthread -O3 -Wall

// In case you experience issues, comment the following line to enable the assert()
#define NDEBUG

#include <iostream>
#include <stdio.h>
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

inline double max(double a, double b) { return a > b ? a : b ; }
inline double min(double a, double b) { return a < b ? a : b ; }


/*********************************/
/*       Tool functions          */
/*********************************/

void save_image(double * image, int height, int width, std::string filename, bool use_log=false)
{
    int i,j;

    double max = image[0];
    double min = image[0];
    for(i = 0 ; i < height ; ++i)
    {
        for(j = 0 ; j < width ; ++j)
        {
            min = min < image[i*width+j] ? min : image[i*width+j];
            max = max > image[i*width+j] ? max : image[i*width+j];
        }
    }

    int * data = new int[width*height];

    if(max != min)
    {
        if(!use_log)
        {
            for(i = 0 ; i < height ; i ++)
            {
                for(j = 0 ; j < width ; j ++)
                {
                    data[i*width + j] = (int)(255.0*(image[i*width+j]-min)/(max-min));
                }
            }
        }
        else
        {
            for(i = 0 ; i < height ; i ++)
            {
                for(j = 0 ; j < width ; j ++)
                {
                    data[i*width + j] = (int)(255.0*(log(fabs(image[i*width+j])+1)/log(1 + fabs(max))));
                }
            }
        }
    }

    cimg_library::CImg<int> img(data,width,height,1);
    img.save(filename.c_str());
    delete[] data;
}

double * setup_filter(std::string filter, int & h_kern, int & w_kern, int kernel_size=3)
{
    double * kernel_array;

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

    return kernel_array;
}

/* ******************************************************************************************************** */
/*          Tool functions to find optimal sizes for the images to convolve with fft and zero-padding       */
/* ******************************************************************************************************** */

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
/*        Linear convolution with FFTW3, computing 2DFT at the same time, and optimizing the sizes          */
/* ******************************************************************************************************** */

void linear_convolution_fft_fftw(double * src, int h_src, int w_src, double * kernel, int h_kernel, int w_kernel, double * dst)
{
    assert(kernel->size1 <= src->size1 && kernel->size2 <= src->size2 && dst->size1 == src->size1 && dst->size2 == dst->size2);

    int height;
    int width;

    height = find_closest_factor(h_src + int((h_kernel+1)/2),FFTW_FACTORS);
    width = find_closest_factor(w_src + int((w_kernel+1)/2), FFTW_FACTORS);

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


/* ******************************************************************************************************** */
/*                   Main : it loads an image and convolves it with a gaussian filter                       */
/* ******************************************************************************************************** */

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
    double * src = new double[h_src* w_src];
    // src is the image to convolve
    for (int i = 0 ; i < h_src ; i++)
        for (int j = 0 ; j  < w_src ; j++)
            src[i*w_src+j] = (int)(image(j,i,0));

    int h_kernel, w_kernel;
    double * kernel = setup_filter("Gaussian", h_kernel, w_kernel, kernel_size);
    double * dst = new double[h_src* w_src];

    // And compute the linear convolution with an optimal size
    gettimeofday(&before, NULL);
    linear_convolution_fft_fftw(src, h_src, w_src,kernel, h_kernel, w_kernel, dst);
    gettimeofday(&after, NULL);
    sbefore = before.tv_sec + before.tv_usec * 1E-6;
    safter =after.tv_sec + after.tv_usec * 1E-6;
    total = safter - sbefore;
    std::cout <<"Linear convolution FFTW, optimal size - Total elapsed time: "<<total <<" s \n";

    save_image(src, h_src, w_src, "src.jpg");
    save_image(kernel, h_kernel, w_kernel,"kernel.jpg");
    save_image(dst,h_src, w_src,"result.jpg");

    // Free the memory
    delete[] src;
    delete[] kernel;
    delete[] dst;
}

