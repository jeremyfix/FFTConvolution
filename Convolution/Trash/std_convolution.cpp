// Compile with g++ -o std_convolution std_convolution.cpp -O3 -Wall

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <cstring>

// To measure the execution time
#include <sys/time.h>

#define NB_REPETITIONS 50

inline double max(double a, double b) { return a > b ? a : b ; }
inline double min(double a, double b) { return a < b ? a : b ; }

/***********************************/
/* Standard circular convolution   */
/***********************************/

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
/*                   Standard  linear convolution                                                           */
/* ******************************************************************************************************** */

void std_linear_convolution(double * src, int h_src, int w_src, double * kernel, int h_kernel, int w_kernel, double * dst)
{
    int i,j,k,l;
    double temp;
    int low_k, high_k, low_l, high_l;

    // For each pixel in the dest image
    for (i = 0 ; i < h_src ; i++)
    {
        low_k = max(0, i + h_kernel/2 - h_src + 1);
        high_k = min(h_kernel, i + h_kernel/2 + 1);

        for (j = 0 ; j  < w_src ; j++)
        {
            low_l = max(0, j + w_kernel/2 - w_src + 1);
            high_l = min(w_kernel, j + w_kernel/2 + 1);
            temp = 0.0;
            // We browse the kernel
            for (k = low_k ; k < high_k ; k++)
            {
                for(l = low_l ; l < high_l ; l++)
                {
                    temp += src[ (i + h_kernel/2 - k)*w_src+ j + w_kernel/2 - l]* kernel[k*w_kernel+l];
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
    if(argc != 3)
    {
        printf("Usage : std_convolution <img_size> <kernel_size>\n");
        printf(" It performs a convolution of an image of size img_size x img_size \n");
        printf(" by a kernel of size kernel_size x kernel_size using nested for loops\n");
        return -1;
    }

    struct timeval before,after;
    double sbefore, safter, total;

    int h_src, w_src, h_kernel, w_kernel;

    h_src = atoi(argv[1]);
    w_src = h_src;
    h_kernel = atoi(argv[2]);
    w_kernel = h_kernel;
    printf("Image of size %i x %i , kernel of size %i x %i \n", h_src, w_src, h_kernel,w_kernel);

    // Build random images to convolve
    double * src = new double[h_src*w_src];
    for(int i = 0 ; i < h_src ; ++i)
        for(int j = 0 ; j < w_src ; ++j)
            src[i*w_src+j]=rand()/double(RAND_MAX);

    double * kernel = new double[h_kernel*w_kernel];
    for(int i = 0 ; i < h_kernel ; ++i)
        for(int j = 0 ; j < w_kernel ; ++j)
            kernel[i*w_kernel+j] = rand()/double(RAND_MAX);

    double * dst = new double[h_src*w_src];
    double * dst_circ = new double[h_src*w_src];

    // And compute the linear convolution
    gettimeofday(&before, NULL);
    for(int i = 0 ; i < NB_REPETITIONS ; ++i)
        std_linear_convolution(src, h_src, w_src, kernel, h_kernel, w_kernel, dst);
    gettimeofday(&after, NULL);
    sbefore = before.tv_sec + before.tv_usec * 1E-6;
    safter =after.tv_sec + after.tv_usec * 1E-6;
    total = safter - sbefore;
    printf("Execution time for a linear convolution : %f s.\n", total/NB_REPETITIONS);

    // And compute the circular convolution
    gettimeofday(&before, NULL);
    for(int i = 0 ; i < NB_REPETITIONS ; ++i)
        std_circular_convolution(src, h_src, w_src, kernel, h_kernel, w_kernel, dst_circ);
    gettimeofday(&after, NULL);
    sbefore = before.tv_sec + before.tv_usec * 1E-6;
    safter =after.tv_sec + after.tv_usec * 1E-6;
    total = safter - sbefore;
    printf("Execution time for a circular convolution : %f s.\n", total/NB_REPETITIONS);

    delete[] src;
    delete[] dst;
    delete[] dst_circ;
    delete[] kernel;
}

