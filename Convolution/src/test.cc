#include <cstdio>
#include "convolution_std.h"
#include "convolution_fftw.h"

void print_tab(double * src, int h, int w)
{
    for(int i = 0 ; i < h ; ++i)
    {
        for(int j = 0 ; j < w ; ++j)
            printf("%i ", int(src[i*w+j]));
        printf("\n");
    }
}

void init(double * src, int h_src, int w_src, double * kernel, int h_kern, int w_kern)
{
    for(int i = 0 ; i < h_src ; ++i)
        for(int j = 0 ; j < w_src ; ++j)
            src[i*w_src+j] = 0.0;
    src[h_src/2 * w_src + w_src/2] = 1.0;

    for(int i = 0 ; i < h_kern ; ++i)
        for(int j = 0 ; j < w_kern ; ++j)
            kernel[i*w_kern + j] = 0.0;
    kernel[2*w_kern+1] = 1.0;
}


int main(int argc, char * argv[])
{
    int h_src = 5;
    int w_src = 5;
    double * src = new double[h_src*w_src];
    int h_kern = 3;
    int w_kern = 3;
    double * kernel = new double[h_kern * w_kern];
    double * res = new double[h_src*w_src];

    init(src, h_src, w_src, kernel, h_kern, w_kern);

    printf("Avant : \n");
    print_tab(src, h_src, w_src);
    printf("\n");
    print_tab(kernel, h_kern, w_kern);
    printf("\n");

    for(int i = 0 ; i <= h_src ; ++i)
    {
        printf("Test : \n");
        print_tab(src, h_src, w_src);
        printf("\n");
        std_circular_convolution(src, h_src, w_src, kernel, h_kern, w_kern, res);
        print_tab(res, h_src, w_src);
        printf("\n");
        for(int i = 0 ; i < h_src ; ++i)
            for(int j = 0 ; j < w_src ; ++j)
                src[i*h_src+j] = res[i*h_src+j];
    }

    // Reinit the ar

}
