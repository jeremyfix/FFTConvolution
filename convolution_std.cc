// g++ -o convolution_std convolution_std.cc -O3
#include <cmath>
#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <algorithm>

typedef enum
{
    STD_LINEAR_FULL,
    STD_LINEAR_SAME,
    STD_LINEAR_VALID,
    STD_CIRCULAR_SAME
} STD_Convolution_Mode;


typedef struct STD_Workspace
{
    int h_src, w_src, h_kernel, w_kernel;
    STD_Convolution_Mode mode;
    double * dst;
    int h_dst, w_dst;
} STD_Workspace;

void init_workspace_std(STD_Workspace & ws, STD_Convolution_Mode mode, int h_src, int w_src, int h_kernel, int w_kernel)
{
    ws.h_src = h_src;
    ws.w_src = w_src;
    ws.h_kernel = h_kernel;
    ws.w_kernel = w_kernel;
    ws.mode = mode;

    switch(mode)
    {
    case STD_LINEAR_FULL:
        ws.h_dst = ws.h_src + ws.h_kernel - 1;
        ws.w_dst = ws.w_src + ws.w_kernel - 1;
        break;
    case STD_LINEAR_SAME:
        ws.h_dst = ws.h_src;
        ws.w_dst = ws.w_src;
        break;
    case STD_LINEAR_VALID:
        ws.h_dst = ws.h_src - ws.h_kernel+1;
        ws.w_dst = ws.w_src - ws.w_kernel+1;
        break;
    case STD_CIRCULAR_SAME:
        ws.h_dst = ws.h_src;
        ws.w_dst = ws.w_src;
        break;
    default:
        printf("Unrecognized mode, valid modes are :\n");
        printf( " STD_LINEAR_FULL \n");
        printf( " STD_LINEAR_SAME \n");
        printf( " STD_LINEAR_VALID \n");
        printf( " STD_CIRCULAR_SAME \n");
    }

    if(ws.h_dst > 0 && ws.w_dst > 0)
        ws.dst = new double[ws.h_dst * ws.w_dst];
    else
        printf("Warning : The result is an empty matrix !\n");
}

void clear_workspace_std(STD_Workspace & ws)
{
    if(ws.h_dst > 0 && ws.w_dst > 0)
        delete[] ws.dst;
}

void std_convolve(STD_Workspace &ws, double * src,double * kernel)
{

    double temp;
    int i,j,k,l;
    int low_k, high_k, low_l, high_l;
    int i_src, j_src;

    if(ws.h_dst <= 0 || ws.w_dst <= 0)
        return;

    switch(ws.mode)
    {
    case STD_LINEAR_FULL:
        // Full linear convolution of size N + M -1
        for( i = 0 ; i < ws.h_dst ; ++i)
        {
            low_k = std::max(0, i - ws.h_kernel + 1);
            high_k = std::min(ws.h_src - 1, i);
            for( j = 0 ; j < ws.w_dst ; ++j)
            {
                low_l = std::max(0, j - ws.w_kernel + 1);
                high_l = std::min(ws.w_src - 1 ,j);
                temp = 0.0;
                for( k = low_k ; k <= high_k ; ++k)
                {
                    for( l = low_l ; l <= high_l ; ++l)
                    {
                        temp += src[k*ws.w_src+l] * kernel[(i-k)*ws.w_kernel + (j-l)];
                    }
                }
                ws.dst[i * ws.w_dst + j] = temp;
            }
        }
        break;
    case STD_LINEAR_SAME:
        // Same linear convolution, of size N
        for( i = 0 ; i < ws.h_dst ; ++i)
        {
            low_k = std::max(0, i - int((ws.h_kernel-1.0)/2.0));
            high_k = std::min(ws.h_src - 1, i + int(ws.h_kernel/2.0));
            for( j = 0 ; j < ws.w_dst ; ++j)
            {
                low_l = std::max(0, j - int((ws.w_kernel-1.0)/2.0));
                high_l = std::min(ws.w_src - 1, j + int(ws.w_kernel/2.0));
                temp = 0.0;
                for( k = low_k ; k <= high_k ; ++k)
                {
                    for( l = low_l ; l <= high_l ; ++l)
                    {
                        temp += src[k*ws.w_src+l] * kernel[(i-k+int(ws.h_kernel/2.0))*ws.w_kernel + (j-l+int(ws.w_kernel/2.0))];
                    }
                }
                ws.dst[i * ws.w_dst + j] = temp;
            }
        }
        break;
    case STD_LINEAR_VALID:
        // Valid linear convolution, of size N - M
        for( i = 0 ; i < ws.h_dst ; ++i)
        {
            for( j = 0 ; j < ws.w_dst ; ++j)
            {
                temp = 0.0;
                for( k = i ; k <= i + ws.h_kernel-1; ++k)
                {
                    for( l = j ; l <= j + ws.w_kernel-1 ; ++l)
                    {
                        temp += src[k*ws.w_src+l] * kernel[(i+ws.h_kernel-1-k)*ws.w_kernel + (j+ws.w_kernel-1-l)];
                    }
                }
                ws.dst[i * ws.w_dst + j] = temp;
            }
        }
        break;
    case STD_CIRCULAR_SAME:
        // Circular convolution, modulo N, shifted by M/2
        // We suppose the filter has a size at most the size of the image
        assert(ws.h_kernel <= ws.h_src && ws.w_kernel <= ws.w_src);

        for (i = 0 ; i < ws.h_dst ; ++i)
        {
            for (j = 0 ; j  < ws.w_dst ; ++j)
            {
                temp = 0.0;
                // We browse the kernel
                for (k = 0 ; k < ws.h_kernel  ; ++k)
                {
                    i_src = i - k + int(ws.h_kernel/2.0);
                    if(i_src < 0)
                        i_src += ws.h_src;
                    else if(i_src >= ws.h_src)
                        i_src -= ws.h_src;

                    for(l = 0 ; l < ws.w_kernel ; ++l)
                    {
                        j_src = j - l + int(ws.w_kernel/2.0);
                        if(j_src < 0)
                            j_src += ws.w_src;
                        else if(j_src >= ws.w_src)
                            j_src -= ws.w_src;

                        temp += src[i_src*ws.w_src + j_src]* kernel[k*ws.w_kernel+l];
                    }
                }
                ws.dst[i*ws.w_src+j] = temp;
            }
        }

        break;
        default:
        break;
    }
}

void print_tab(double * tab, int width, int height)
{
    printf("Tab : \n");
    for(int i = 0 ; i < height ; ++i)
    {
        for(int j = 0 ; j < width ; ++j)
        {
            printf(" %i ", int(tab[i*width+j]));
        }
        printf("\n");
    }
}

int main(int argc, char * argv[])
{
    int Ns = atoi(argv[1]);
    int Nk = atoi(argv[2]);

    // Create a small source image
    // with a 1 at the beginning
    double *src = new double[Ns];
    for(int i = 0 ; i < Ns; ++i)
        src[i] = rand()/double(RAND_MAX);

    double *kernel = new double[Nk];
    for(int i = 0 ; i < Nk ; ++i)
        kernel[i] = rand()/double(RAND_MAX);

    // Let's perform some linear convolutions
    STD_Workspace ws;
    init_workspace_std(ws, STD_LINEAR_FULL, 1, Ns, 1, Nk);

    std_convolve(ws, src, kernel);
    printf("c = [");
    for(int i = 0 ; i < ws.w_dst ; ++i)
            printf(" %f ", ws.dst[i]);
    printf(" ]\n");

    // Print the matlab command for testing
    printf("Matlab command : \n");
    printf("f=[");
    for(int i =0 ; i < Ns-1 ; ++i)
        printf("%f,", src[i]);
    printf("%f];",src[Ns-1]);
    printf("g=[");
    for(int i =0 ; i < Nk-1 ; ++i)
        printf("%f,", kernel[i]);
    printf("%f];", kernel[Nk-1]);
    printf("conv(f,g,'full')\n");

    clear_workspace_std(ws);

    // Same
    printf("\n\n");
    init_workspace_std(ws, STD_LINEAR_SAME, 1, Ns, 1, Nk);

    std_convolve(ws, src, kernel);
    printf("c = [");
    for(int j = 0 ; j < ws.w_dst ; ++j)
            printf(" %f ", ws.dst[j]);
    printf(" ]\n");

    // Print the matlab command for testing
    printf("Matlab command : \n");
    printf("f=[");
    for(int i =0 ; i < Ns-1 ; ++i)
        printf("%f,", src[i]);
    printf("%f];",src[Ns-1]);
    printf("g=[");
    for(int i =0 ; i < Nk-1 ; ++i)
        printf("%f,", kernel[i]);
    printf("%f];", kernel[Nk-1]);
    printf("conv(f,g,'same')\n");

    clear_workspace_std(ws);

    // Valid
    printf("\n\n");
    init_workspace_std(ws, STD_LINEAR_VALID, 1, Ns, 1, Nk);

    std_convolve(ws, src, kernel);
    printf("c = [");
    for(int j = 0 ; j < ws.w_dst ; ++j)
            printf(" %f ", ws.dst[j]);
    printf(" ]\n");

    // Print the matlab command for testing
    printf("Matlab command : \n");
    printf("f=[");
    for(int i =0 ; i < Ns-1 ; ++i)
        printf("%f,", src[i]);
    printf("%f];",src[Ns-1]);
    printf("g=[");
    for(int i =0 ; i < Nk-1 ; ++i)
        printf("%f,", kernel[i]);
    printf("%f];", kernel[Nk-1]);
    printf("conv(f,g,'valid')\n");

    clear_workspace_std(ws);

    // Circular
    printf("\n\n");
    init_workspace_std(ws, STD_CIRCULAR_SAME, 1, Ns, 1, Nk);

    std_convolve(ws, src, kernel);
    printf("c = [");
    for(int j = 0 ; j < ws.w_dst ; ++j)
            printf(" %f ", ws.dst[j]);
    printf(" ]\n");

    // Print the matlab command for testing
    printf("Matlab command : \n");
    printf("f=[");
    for(int i =0 ; i < Ns-1 ; ++i)
        printf("%f,", src[i]);
    printf("%f];",src[Ns-1]);
    printf("g=[");
    for(int i =0 ; i < Nk-1 ; ++i)
        printf("%f,", kernel[i]);
    printf("%f];", kernel[Nk-1]);
    printf("cconv(f,g,%i)\n", Ns);

    clear_workspace_std(ws);

    delete[] src;
    delete[] kernel;
}



//int main(int argc, char * argv[])
//{
//    int N = 6;
//
//    // Create a small source image
//    // with a 1 at the beginning
//    double *src = new double[N];
//    for(int i = 0 ; i < N; ++i)
//        src[i] = 0;
//    src[0] = 1;
//
//    // Create a kernel with a 1 just at the right of the center
//    // this shifts the input 1 pixel to the right
//    double *kernel = new double[N];
//    for(int i = 0 ; i < N ; ++i)
//        kernel[i] = 0;
//    kernel[int(N/2)+1] = 1;
//
//    // Print the arrays
//    printf("Source : ");
//    for(int i = 0 ; i < N ; ++i)
//    {
//            printf(" %i ", int(src[i]));
//    }
//    printf("\n");
//
//    printf(" Center of the kernel at index %i, starting from 0 \n", int(N/2));
//    printf("Kernel : ");
//    for(int i = 0 ; i < N ; ++i)
//    {
//        printf(" %i ", int(kernel[i]));
//    }
//    printf("\n");
//
//
//    // Let's perform some linear convolutions
//    STD_Workspace ws;
//    init_workspace_std(ws, STD_LINEAR_SAME, 1, N, 1, N);
//
//    printf("\n");
//    printf("Linear convolutions : \n");
//    for(int i = 1 ; i <= N+2 ; ++i)
//    {
//        std_convolve(ws, src, kernel);
//
//        printf("#%i : ", i);
//        for(int j = 0 ; j < ws.w_dst ; ++j)
//            printf(" %.0f ", fabs(ws.dst[j]) < 1e-10 ? 0.0 : ws.dst[j]);
//        printf("\n");
//
//        // Copy ws.dst in src
//        for(int j = 0 ; j < N ; ++j)
//            src[j] = ws.dst[j];
//    }
//    clear_workspace_std(ws);
//
//    printf("\n\n");
//
//    // **************
//
//    // Create a small source image
//    // with a 1 at the beginning
//    for(int i = 0 ; i < N; ++i)
//        src[i] = 0;
//    src[0] = 1;
//
//    // Create a kernel with a 1 just at the right of the center
//    // this shifts the input 1 pixel to the right
//    for(int i = 0 ; i < N ; ++i)
//        kernel[i] = 0;
//    kernel[int(N/2)+1] = 1;
//
//    // Print the arrays
//    printf("Source : ");
//    for(int i = 0 ; i < N ; ++i)
//    {
//            printf(" %i ", int(src[i]));
//    }
//    printf("\n");
//
//    printf(" Center of the kernel at index %i, starting from 0 \n", int(N/2));
//    printf("Kernel : ");
//    for(int i = 0 ; i < N ; ++i)
//    {
//        printf(" %i ", int(kernel[i]));
//    }
//    printf("\n");
//
//    // Let's perform some circular convolutions
//    init_workspace_std(ws, STD_CIRCULAR_SAME, 1, N, 1, N);
//
//    printf("\n");
//    printf("Circular convolutions : \n");
//    for(int i = 1 ; i <= N+2 ; ++i)
//    {
//        std_convolve(ws, src, kernel);
//        printf("#%i : ", i);
//        for(int j = 0 ; j < ws.w_dst ; ++j)
//            printf(" %.0f ", fabs(ws.dst[j]) < 1e-10 ? 0.0 : ws.dst[j]);
//        printf("\n");
//        // Copy ws.dst in src
//        for(int j = 0 ; j < N ; ++j)
//            src[j] = ws.dst[j];
//    }
//
//    clear_workspace_std(ws);
//
//    delete[] src;
//    delete[] kernel;
//}
