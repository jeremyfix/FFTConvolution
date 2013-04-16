#ifndef CONVOLUTION_STD_H
#define CONVOLUTION_STD_H

#include <cmath>

#ifndef max_func
#define max_func
inline double max(double a, double b) { return a > b ? a : b ; }
#endif

#ifndef min_func
#define min_func
inline double min(double a, double b) { return a < b ? a : b ; }
#endif

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
                i_src = i - k;
                if(i_src < 0)
                    i_src += h_src;
                else if(i_src >= h_src)
                    i_src -= h_src;

                for(l = 0 ; l < w_kernel ; ++l)
                {
                    j_src = j - l;
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
/*     Standard  linear convolution, returns a result of the same size as the original src signal           */
/* ******************************************************************************************************** */

void std_linear_convolution(double * src, int h_src, int w_src, double * kernel, int h_kernel, int w_kernel, double * dst)
{
    int i,j,k,l;
    double temp;
    int low_k, high_k, low_l, high_l;

    // For each pixel in the dest image
    for (i = 0 ; i < h_src ; i++)
    {
        low_k = max(0, i - int((h_kernel-1)/2));
        high_k = min(h_src-1, i + int(h_kernel/2));

        for (j = 0 ; j  < w_src ; j++)
        {
            low_l = max(0, j - int((w_kernel-1)/2));
            high_l = min(w_src-1, j + int(w_kernel/2));
            temp = 0.0;
            // We browse the kernel
            for (k = low_k ; k <= high_k ; k++)
            {
                for(l = low_l ; l <= high_l ; l++)
                {
                    temp += src[k*w_src+l]* kernel[(i+int(h_kernel/2)-k)*w_kernel+(j+int(w_kernel/2)-l)];
                }
            }
            dst[i*w_src+j] = temp;
        }
    }
}

#endif
