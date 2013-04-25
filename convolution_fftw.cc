// g++ -o convolution_fftw convolution_fftw.cc `pkg-config --libs --cflags fftw3` -O3

#include <cassert>
#include <fftw3.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <algorithm>

// ******************** Begin of factorization code ***********************//
// A piece of code to determine if a number "n" can be written as products 
// of only the integers given in implemented_factors
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
      return ;
    }

  if (n == 1)
    {
      factors[0] = 1;
      *n_factors = 1;
      return ;
    }

  /* deal with the implemented factors */

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
  // We check that n is not a multiple of 4*4*4*2
  if(n % 4*4*4*2 == 0)
    return false;

  int nf;
  int factors[64];
  int i = 0;
  factorize(n, &nf, factors,implemented_factors);

  // We just have to check if the last factor belongs to GSL_FACTORS
  while(implemented_factors[i])
    {
      if(factors[nf-1] == implemented_factors[i])
	return true;
      ++i;
    }
  return false;
}

int find_closest_factor(int n, int * implemented_factor)
{
  int j;
  if(is_optimal(n,implemented_factor))
    return n;
  else
    {
      j = n+1;
      while(!is_optimal(j,implemented_factor))
	++j;
      return j;
    }
}
// ******************** End of factorization code ***********************//

int FFTW_FACTORS[7] = {13,11,7,5,3,2,0}; // end with zero to detect the end of the array

typedef enum
  {
    FFTW_LINEAR_FULL,
    FFTW_LINEAR_SAME,
    FFTW_LINEAR_VALID,
    FFTW_CIRCULAR_SAME
  } FFTW_Convolution_Mode;

typedef struct FFTW_Workspace
{
  fftw_complex * in_src, *out_src;
  int h_src, w_src, h_kernel, w_kernel;
  int w_fftw, h_fftw;
  fftw_plan p_forw;
  fftw_plan p_back;
  FFTW_Convolution_Mode mode;
  double * dst; // The array containing the result
  int h_dst; // its height
  int w_dst; // its width
} FFTW_Workspace;

void init_workspace_fftw(FFTW_Workspace & ws, FFTW_Convolution_Mode mode, int h_src, int w_src, int h_kernel, int w_kernel)
{
  ws.h_src = h_src;
  ws.w_src = w_src;
  ws.h_kernel = h_kernel;
  ws.w_kernel = w_kernel;
  ws.mode = mode;

  switch(mode)
    {
    case FFTW_LINEAR_FULL:
      // Full Linear convolution
      ws.h_fftw = find_closest_factor(h_src + h_kernel - 1,FFTW_FACTORS);
      ws.w_fftw = find_closest_factor(w_src + w_kernel - 1,FFTW_FACTORS);
      ws.h_dst = h_src + h_kernel-1;
      ws.w_dst = w_src + w_kernel-1;
      break;
    case FFTW_LINEAR_SAME:
      // Same Linear convolution
      ws.h_fftw = find_closest_factor(h_src + int(h_kernel/2.0),FFTW_FACTORS);
      ws.w_fftw = find_closest_factor(w_src + int(w_kernel/2.0),FFTW_FACTORS);
      ws.h_dst = h_src;
      ws.w_dst = w_src;
      break;
    case FFTW_LINEAR_VALID:
      // Valid Linear convolution
      if(ws.h_kernel > ws.h_src || ws.w_kernel > ws.w_src)
        {
	  printf("Warning : The 'valid' convolution results in an empty matrix\n");
	  ws.h_fftw = 0;
	  ws.w_fftw = 0;
	  ws.h_dst = 0;
	  ws.w_dst = 0;
        }
      else
        {
	  ws.h_fftw = find_closest_factor(h_src, FFTW_FACTORS);
	  ws.w_fftw = find_closest_factor(w_src, FFTW_FACTORS);
	  ws.h_dst = h_src - h_kernel+1;
	  ws.w_dst = w_src - w_kernel+1;
        }
      break;
    case FFTW_CIRCULAR_SAME:
      // Circular convolution, modulo N, shifted by M/2
      // We don't padd with zeros because if we do, we need to padd with at least h_kernel/2; w_kernel/2 elements
      // plus the wrapp around
      // which in facts leads to too much computations compared to the gain obtained with the optimal size
      ws.h_fftw = h_src;
      ws.w_fftw = w_src;
      ws.h_dst = h_src;
      ws.w_dst = w_src;
      break;
    default:
      printf("Unrecognized convolution mode, possible modes are :\n");
      printf("   - FFTW_LINEAR_FULL \n");
      printf("   - FFTW_LINEAR_SAME \n");
      printf("   - FFTW_LINEAR_VALID \n");
      printf("   - FFTW_CIRCULAR_SAME \n");
      // TODO EXCEPTION
    }
  printf("Size of FFTW : %i %i \n", ws.h_fftw, ws.w_fftw);
  ws.in_src = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * ws.h_fftw * ws.w_fftw);
  ws.out_src = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * ws.h_fftw * ws.w_fftw);
  ws.p_forw = fftw_plan_dft_2d(ws.h_fftw, ws.w_fftw, ws.in_src, ws.out_src, FFTW_FORWARD, FFTW_ESTIMATE);
  ws.p_back = fftw_plan_dft_2d(ws.h_fftw, ws.w_fftw, ws.in_src, ws.out_src, FFTW_BACKWARD, FFTW_ESTIMATE);

  ws.dst = new double[ws.h_dst * ws.w_dst];
}

void clear_workspace_fftw(FFTW_Workspace & ws)
{
  fftw_destroy_plan(ws.p_forw);
  fftw_destroy_plan(ws.p_back);
  fftw_free(ws.in_src);
  fftw_free(ws.out_src);

  delete[] ws.dst;
}


void fftw_convolve(FFTW_Workspace &ws, double * src,double * kernel)
{
  // First clean up in_src;
  for(int i = 0 ; i < ws.h_fftw ; ++i)
    {
      for(int j = 0 ; j < ws.w_fftw ; ++j)
        {
	  ws.in_src[i*ws.w_fftw+j][0] = 0.0;
	  ws.in_src[i*ws.w_fftw+j][1] = 0.0;
        }
    }

  // Copy the src and kernel arrays in in_src
  // with respectively :
  //            src in the real part of in_src
  //         kernel in the imaginary part of in_src

  int i_fftw, j_fftw;
  int min_h, max_h, min_w,max_w;
  switch(ws.mode)
    {
    case FFTW_LINEAR_FULL:
      for(int i = 0 ; i < ws.h_src ; ++i)
	for(int j = 0 ; j < ws.w_src ; ++j)
	  ws.in_src[i*ws.w_fftw+j][0] = src[i*ws.w_src + j];
      for(int i = 0 ; i < ws.h_kernel ; ++i)
	for(int j = 0 ; j < ws.w_kernel ; ++j)
	  ws.in_src[i*ws.w_fftw+j][1] = kernel[i*ws.w_kernel+j];
      break;
    case FFTW_LINEAR_SAME:
      for(int i = 0 ; i < ws.h_src ; ++i)
	for(int j = 0 ; j < ws.w_src ; ++j)
	  ws.in_src[i*ws.w_fftw+j][0] = src[i*ws.w_src + j];
      min_w = std::max(0, int((ws.w_kernel-1.0)/2.0)-ws.w_src);
      max_w = std::min(ws.w_fftw, ws.w_kernel);
      min_h = std::max(0, int((ws.h_kernel-1.0)/2.0)-ws.h_src);
      max_h = std::min(ws.h_fftw, ws.h_kernel);
      for(int i = min_h ; i < max_h ; ++i)
	for(int j = min_w ; j < max_w ; ++j)
	  ws.in_src[i*ws.w_fftw+j][1] = kernel[i*ws.w_kernel+j];
      break;
    case FFTW_LINEAR_VALID:
      for(int i = 0 ; i < ws.h_src ; ++i)
	for(int j = 0 ; j < ws.w_src ; ++j)
	  ws.in_src[i*ws.w_fftw+j][0] = src[i*ws.w_src + j];
      for(int i = 0 ; i < ws.h_kernel ; ++i)
	for(int j = 0 ; j < ws.w_kernel ; ++j)
	  ws.in_src[i*ws.w_fftw+j][1] = kernel[i*ws.w_kernel+j];
      break;
    case FFTW_CIRCULAR_SAME:
      for(int i = 0 ; i < ws.h_src ; ++i)
	for(int j = 0 ; j < ws.w_src ; ++j)
	  ws.in_src[i*ws.w_fftw+j][0] = src[i*ws.w_src + j];
      for(int i = 0 ; i < ws.h_kernel ; ++i)
        {
	  for(int j = 0 ; j < ws.w_kernel ; ++j)
            {
	      /*i_fftw = i-int(ws.h_kernel/2.0);
                if(i_fftw < 0)
		i_fftw += ws.h_fftw;
                j_fftw = j -int(ws.w_kernel/2.0);
                if(j_fftw < 0)
		j_fftw += ws.w_fftw;*/
	      i_fftw = i % ws.h_fftw;
	      j_fftw = j % ws.w_fftw;
	      ws.in_src[i_fftw*ws.w_fftw+j_fftw][1] = ws.in_src[i_fftw*ws.w_fftw+j_fftw][1]+kernel[i*ws.w_kernel + j];
            }
        }
      break;
    default:
      break;
    }

  // And we now compute the circular convolution of src_P and kernel_P
  // Compute the forward fft
  fftw_execute(ws.p_forw);

  double re_h, im_h, re_hs, im_hs;
  // Compute the element-wise product
  for(int i = 0 ; i < ws.h_fftw ; ++i)
    {
      for(int j = 0 ; j < ws.w_fftw ; ++j)
        {
	  re_h = ws.out_src[i*ws.w_fftw+ j][0];
	  im_h = ws.out_src[i*ws.w_fftw+ j][1];
	  re_hs = ws.out_src[((ws.h_fftw-i)%ws.h_fftw)*ws.w_fftw + (ws.w_fftw-j)%ws.w_fftw][0];
	  im_hs = - ws.out_src[((ws.h_fftw-i)%ws.h_fftw)*ws.w_fftw + (ws.w_fftw-j)%ws.w_fftw][1];

	  ws.in_src[i*ws.w_fftw+j][0] =  0.5*(re_h*im_h - re_hs*im_hs);
	  ws.in_src[i*ws.w_fftw+j][1] = -0.25*(re_h*re_h - im_h * im_h - re_hs * re_hs + im_hs * im_hs);
        }
    }

  // Compute the backward fft
  fftw_execute(ws.p_back);

  // Depending on the type of convolution one is looking for, we extract the appropriate part of the result from out_src
  int h_offset, w_offset;
  switch(ws.mode)
    {
    case FFTW_LINEAR_FULL:
      // Full Linear convolution
      // Here we just keep the first [0:h_dst-1 ; 0:w_dst-1] real part elements of out_src
      for(int i = 0 ; i < ws.h_dst ; ++i)
	for(int j = 0 ; j < ws.w_dst ; ++j)
	  ws.dst[i*ws.w_dst + j] = ws.out_src[i*ws.w_fftw+j][0]/double(ws.w_fftw * ws.h_fftw);
      break;
    case FFTW_LINEAR_SAME:
      // Same linear convolution
      // Here we just keep the first [h_filt/2:h_filt/2+h_dst-1 ; w_filt/2:w_filt/2+w_dst-1] real part elements of out_src
      h_offset = int(ws.h_kernel/2.0);
      w_offset = int(ws.w_kernel/2.0);
      for(int i = 0 ; i < ws.h_dst ; ++i)
	for(int j = 0 ; j < ws.w_dst ; ++j)
	  ws.dst[i*ws.w_dst + j] = ws.out_src[(i+h_offset)*ws.w_fftw+j+w_offset][0]/double(ws.w_fftw * ws.h_fftw);
      break;
    case FFTW_LINEAR_VALID:
      // Valid linear convolution
      // Here we just take [h_dst x w_dst] elements starting at [h_kernel-1;w_kernel-1]
      h_offset = ws.h_kernel - 1;
      w_offset = ws.w_kernel - 1;
      for(int i = 0 ; i < ws.h_dst ; ++i)
	for(int j = 0  ; j < ws.w_dst ; ++j)
	  ws.dst[i*ws.w_dst +j] = ws.out_src[(i+h_offset)*ws.w_fftw+j+w_offset][0]/double(ws.w_fftw*ws.h_fftw);
      break;
    case FFTW_CIRCULAR_SAME:
      // Circular convolution
      // We copy the first [0:h_dst-1 ; 0:w_dst-1] real part elements of out_src
      for(int i = 0 ; i < ws.h_dst ; ++i)
	for(int j = 0 ; j < ws.w_dst ; ++j)
	  ws.dst[i*ws.w_dst + j] = ws.out_src[i*ws.w_fftw+j][0]/double(ws.w_fftw * ws.h_fftw);
      break;
    default:
      printf("Unrecognized convolution mode, possible modes are :\n");
      printf("   - FFTW_LINEAR_FULL \n");
      printf("   - FFTW_LINEAR_SAME \n");
      printf("   - FFTW_LINEAR_VALID \n");
      printf("   - FFTW_CIRCULAR_SAME \n");
    }
}

int main(int argc, char * argv[])
{
  if(argc != 3)
    {
      std::cerr << "Usage : " << argv[0] << " <source_size> <kernel_size>" << std::endl;
      return -1;
    }

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
  FFTW_Workspace ws;
  init_workspace_fftw(ws, FFTW_LINEAR_FULL, 1, Ns, 1, Nk);

  fftw_convolve(ws, src, kernel);
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
  printf("convn(f,g,'full')\n");

  clear_workspace_fftw(ws);

  // Same
  printf("\n\n");
  init_workspace_fftw(ws, FFTW_LINEAR_SAME, 1, Ns, 1, Nk);

  fftw_convolve(ws, src, kernel);
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
  printf("convn(f,g,'same')\n");

  clear_workspace_fftw(ws);

  // Valid
  printf("\n\n");
  init_workspace_fftw(ws, FFTW_LINEAR_VALID, 1, Ns, 1, Nk);

  fftw_convolve(ws, src, kernel);
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
  printf("convn(f,g,'valid')\n");

  clear_workspace_fftw(ws);


  // Circular same
  printf("\n\n");
  init_workspace_fftw(ws, FFTW_CIRCULAR_SAME, 1, Ns, 1, Nk);

  fftw_convolve(ws, src, kernel);
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

  clear_workspace_fftw(ws);


  delete[] src;
  delete[] kernel;
}


//int main(int argc, char * argv[])
//{
//    int N = 6;
//
//    // ************** LINEAR CONVOLUTIONS
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
//        printf(" %i ", int(src[i]));
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
//    FFTW_Workspace ws;
//    init_workspace_fftw(ws, FFTW_LINEAR_SAME, 1, N, 1, N);
//
//    printf("\n");
//    printf("Linear convolutions : \n");
//    for(int i = 1 ; i <= N+2 ; ++i)
//    {
//        fftw_convolve(ws, src, kernel);
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
//    clear_workspace_fftw(ws);
//
//    printf("\n\n");
//
//    // ************** CIRCULAR CONVOLUTIONS
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
//        printf(" %i ", int(src[i]));
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
//    init_workspace_fftw(ws, FFTW_CIRCULAR_SAME, 1, N, 1, N);
//
//    printf("\n");
//    printf("Circular convolutions : \n");
//    for(int i = 1 ; i <= N+2 ; ++i)
//    {
//        fftw_convolve(ws, src, kernel);
//        printf("#%i : ", i);
//        for(int j = 0 ; j < ws.w_dst ; ++j)
//            printf(" %.0f ", fabs(ws.dst[j]) < 1e-10 ? 0.0 : ws.dst[j]);
//        printf("\n");
//        // Copy ws.dst in src
//        for(int j = 0 ; j < N ; ++j)
//            src[j] = ws.dst[j];
//    }
//
//    clear_workspace_fftw(ws);
//
//    delete[] src;
//    delete[] kernel;
//}
