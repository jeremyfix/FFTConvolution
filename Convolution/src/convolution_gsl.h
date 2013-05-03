#ifndef CONVOLUTION_GSL_H
#define CONVOLUTION_GSL_H

#include "factorize.h"
#include <cassert>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_fft_complex.h>

namespace GSL_Convolution 
{

  int GSL_FACTORS[7] = {7,6,5,4,3,2,0}; // end with zero to detect the end of the array

  typedef enum
  {
    LINEAR_FULL,
    LINEAR_SAME_UNPADDED,
    LINEAR_SAME,
    LINEAR_VALID,
    CIRCULAR_SAME,
    CIRCULAR_SAME_PADDED,
    CIRCULAR_FULL_UNPADDED,
    CIRCULAR_FULL
  } Convolution_Mode;
  

  typedef struct Workspace
  {
    gsl_fft_complex_workspace *ws_column, *ws_line;
    gsl_fft_complex_wavetable *wv_column, *wv_line;
    int h_src, w_src, h_kernel, w_kernel;
    int h_fft, w_fft;
    gsl_matrix *fft, *fft_copy;
    Convolution_Mode mode;
    double * dst; // The array containing the result
    int h_dst; // its height
    int w_dst; // its width
  } Workspace;

  void init_workspace(Workspace & ws, Convolution_Mode mode, int h_src, int w_src, int h_kernel, int w_kernel)
  {
    ws.h_src = h_src;
    ws.w_src = w_src;
    ws.h_kernel = h_kernel;
    ws.w_kernel = w_kernel;
    ws.mode = mode;

    switch(mode)
      {
      case LINEAR_FULL:
	// Full Linear convolution
	ws.h_fft = find_closest_factor(h_src + h_kernel - 1,GSL_FACTORS);
	ws.w_fft = find_closest_factor(w_src + w_kernel - 1,GSL_FACTORS);
	ws.h_dst = h_src + h_kernel-1;
	ws.w_dst = w_src + w_kernel-1;
	break;
      case LINEAR_SAME_UNPADDED:
	// Same Linear convolution
	ws.h_fft = h_src + int(h_kernel/2.0);
	ws.w_fft = w_src + int(w_kernel/2.0);
	ws.h_dst = h_src;
	ws.w_dst = w_src;
	break;
      case LINEAR_SAME:
	// Same Linear convolution
	ws.h_fft = find_closest_factor(h_src + int(h_kernel/2.0),GSL_FACTORS);
	ws.w_fft = find_closest_factor(w_src + int(w_kernel/2.0),GSL_FACTORS);
	ws.h_dst = h_src;
	ws.w_dst = w_src;
	break;
      case LINEAR_VALID:
	// Valid Linear convolution
	if(ws.h_kernel > ws.h_src || ws.w_kernel > ws.w_src)
	  {
	    //printf("Warning : The 'valid' convolution results in an empty matrix\n");
	    ws.h_fft = 0;
	    ws.w_fft = 0;
	    ws.h_dst = 0;
	    ws.w_dst = 0;
	  }
	else
	  {
	    ws.h_fft = find_closest_factor(h_src, GSL_FACTORS);
	    ws.w_fft = find_closest_factor(w_src, GSL_FACTORS);
	    ws.h_dst = h_src - h_kernel+1;
	    ws.w_dst = w_src - w_kernel+1;
	  }
	break;	
      case CIRCULAR_SAME:
	// Circular convolution
	ws.h_fft = h_src;
	ws.w_fft = w_src;
	ws.h_dst = h_src;
	ws.w_dst = w_src;
	break;
      case CIRCULAR_SAME_PADDED:
	// Cicular convolution with optimal sizes
	ws.h_fft = find_closest_factor(h_src+h_kernel, GSL_FACTORS);
	ws.w_fft = find_closest_factor(w_src+w_kernel, GSL_FACTORS);
	ws.h_dst = h_src;
	ws.w_dst = w_src;
	break;
      case CIRCULAR_FULL_UNPADDED:
        // We here want to compute a circular convolution modulo h_dst, w_dst
        // These two variables must have been set before calling init_workscape !!
	ws.h_dst = h_src + h_kernel - 1;
	ws.w_dst = w_src + w_kernel - 1;
	ws.h_fft = find_closest_factor(h_src + h_kernel - 1, GSL_FACTORS);
	ws.w_fft = find_closest_factor(w_src + w_kernel - 1, GSL_FACTORS);
        break;
      case CIRCULAR_FULL:
        // We here want to compute a circular convolution modulo h_dst, w_dst
        // These two variables must have been set before calling init_workscape !!
        ws.h_dst = h_src + h_kernel - 1;
        ws.w_dst = w_src + w_kernel - 1;
        ws.h_fft = ws.h_dst;
        ws.w_fft = ws.w_dst;
        break;
      default:
	printf("Unrecognized convolution mode, possible modes are :\n");
	printf("   - LINEAR_FULL \n");
	printf("   - LINEAR_SAME \n");
	printf("   - LINEAR_SAME_UNPADDED \n");
	printf("   - LINEAR_VALID \n");
        printf("   - CIRCULAR_SAME \n");
        printf("   - CIRCULAR_SAME_PADDED \n");
        printf("   - CIRCULAR_FULL_UNPADDED\n");
        printf("   - CIRCULAR_FULL\n");
      }

    // We need to create a matrix holding 2 times the number of coefficients of src for the real and imaginary parts
    if(ws.h_fft != 0 && ws.w_fft != 0)
      {
	ws.fft = gsl_matrix_alloc(ws.h_fft, 2*ws.w_fft);
	ws.fft_copy = gsl_matrix_alloc(ws.h_fft, 2*ws.w_fft);
	
	ws.ws_column = gsl_fft_complex_workspace_alloc(ws.h_fft);
	ws.ws_line = gsl_fft_complex_workspace_alloc(ws.w_fft);
	ws.wv_column = gsl_fft_complex_wavetable_alloc(ws.h_fft);
	ws.wv_line = gsl_fft_complex_wavetable_alloc(ws.w_fft);
      }
    ws.dst = new double[ws.h_dst * ws.w_dst];
  }

  void clear_workspace(Workspace & ws)
  {
    if(ws.h_fft != 0 && ws.w_fft != 0)
      {
	gsl_fft_complex_workspace_free(ws.ws_column);
	gsl_fft_complex_workspace_free(ws.ws_line);
	gsl_fft_complex_wavetable_free(ws.wv_column);
	gsl_fft_complex_wavetable_free(ws.wv_line);

	gsl_matrix_free(ws.fft);
	gsl_matrix_free(ws.fft_copy);
      }
    delete[] ws.dst;
  }

  // Compute the circular convolution of src and kernel modulo ws.h_fft, ws.w_fft
  // using the Fast Fourier Transform
  // The result is in ws.dst
  void gsl_circular_convolution(Workspace &ws, double * src, double * kernel)
  {

    // Reset the content of ws.fft
    gsl_matrix_set_zero(ws.fft);
    
    // Then we build our periodic signals
    // src is copied in the real part
    for(int i = 0 ; i < ws.h_src ; ++i)
      for(int j = 0 ; j < ws.w_src ; ++j)
	ws.fft->data[(i%ws.h_fft)*2*ws.w_fft+(j%ws.w_fft)*2] += src[i*ws.w_src + j];
    // kernel is copied in the imaginary part
    for(int i = 0 ; i < ws.h_kernel ; ++i)
      for(int j = 0 ; j < ws.w_kernel ; ++j)
	ws.fft->data[(i%ws.h_fft)*2*ws.w_fft+(j%ws.w_fft)*2 + 1] += kernel[i*ws.w_kernel + j];

    // We compute the 2 forward DFT at once
    // Apply the FFT on the line i
    for(int i = 0 ; i < ws.h_fft ; ++i)
      gsl_fft_complex_forward (&ws.fft->data[i*2*ws.w_fft],1, ws.w_fft, ws.wv_line, ws.ws_line);
    
    // Apply the FFT on the column j
    for(int j = 0 ; j < ws.w_fft ; ++j)
      gsl_fft_complex_forward (&ws.fft->data[2*j], ws.w_fft, ws.h_fft, ws.wv_column, ws.ws_column);

    // Their element-wise product : be carefull, the matrices hold complex numbers !
    // We need a copy of fft to perform the product properly
    double re_h, im_h, re_hs, im_hs;
    for(int i = 0 ; i < ws.h_fft ; ++i)
      {
  	for(int j = 0 ; j < ws.w_fft ; ++j)
  	  {
  	    re_h = ws.fft->data[i*2*ws.w_fft + 2*j];
  	    im_h = ws.fft->data[i*2*ws.w_fft + 2*j+1];
  	    re_hs = ws.fft->data[((ws.h_fft-i)%ws.h_fft)*2*ws.w_fft + ((2*ws.w_fft-2*j)%(2*ws.w_fft))];
  	    im_hs = -ws.fft->data[((ws.h_fft-i)%ws.h_fft)*2*ws.w_fft + ((2*ws.w_fft-2*j)%(2*ws.w_fft))+1];

  	    ws.fft_copy->data[i*2*ws.w_fft+2*j] = 0.5*(re_h*im_h - re_hs*im_hs);
  	    ws.fft_copy->data[i*2*ws.w_fft+2*j+1] = -0.25*(re_h*re_h - im_h * im_h - re_hs * re_hs + im_hs * im_hs);
  	  }
      }

    // And the inverse FFT, which is done in the similar way as before
    for(int i = 0 ; i < ws.h_fft ; ++i)
      {
  	// Apply the FFT^{-1} on the line i
  	gsl_fft_complex_inverse(&ws.fft_copy->data[i*2*ws.w_fft],1, ws.w_fft, ws.wv_line, ws.ws_line);
      }

    for(int j = 0 ; j < ws.w_fft ; ++j)
      {
  	// Apply the FFT^{-1} on the column j
  	gsl_fft_complex_inverse(&ws.fft_copy->data[2*j], ws.w_fft, ws.h_fft, ws.wv_column, ws.ws_column);
      }


  }

  void convolve(Workspace &ws, double * src,double * kernel)
  {
    if(ws.h_fft <= 0 || ws.w_fft <= 0)
      return;

    // Compute the circular convolution
    gsl_circular_convolution(ws, src, kernel);

    // Depending on the type of convolution one is looking for, we extract the appropriate part of the result from out_src
    int h_offset, w_offset;

    switch(ws.mode)
      {
      case LINEAR_FULL:
        // Full Linear convolution
        // Here we just keep the first [0:h_dst-1 ; 0:w_dst-1] real part elements of out_src
        for(int i = 0 ; i < ws.h_dst  ; ++i)
	  for(int j = 0 ; j < ws.w_dst ; ++j)
	    ws.dst[i*ws.w_dst+j] = ws.fft_copy->data[i*2*ws.w_fft+2*j];
        break;
      case LINEAR_SAME_UNPADDED:
      case LINEAR_SAME:
        // Same linear convolution
        // Here we just keep the first [h_filt/2:h_filt/2+h_dst-1 ; w_filt/2:w_filt/2+w_dst-1] real part elements of out_src
        h_offset = int(ws.h_kernel/2.0);
        w_offset = int(ws.w_kernel/2.0);
        for(int i = 0 ; i < ws.h_dst ; ++i)
	  for(int j = 0 ; j < ws.w_dst ; ++j)
	    ws.dst[i*ws.w_dst+j] = ws.fft_copy->data[(i+h_offset)*2*ws.w_fft+2*(j+w_offset)];
        break;
      case LINEAR_VALID:
        // Valid linear convolution
        // Here we just take [h_dst x w_dst] elements starting at [h_kernel-1;w_kernel-1]
        h_offset = ws.h_kernel - 1;
        w_offset = ws.w_kernel - 1;
        for(int i = 0 ; i < ws.h_dst ; ++i)
	  for(int j = 0 ; j < ws.w_dst ; ++j)
	    ws.dst[i*ws.w_dst+j] = ws.fft_copy->data[(i+h_offset)*2*ws.w_fft+2*(j+w_offset)];
        break;
      case CIRCULAR_SAME:
      case CIRCULAR_FULL:
      case CIRCULAR_SAME_PADDED:
      case CIRCULAR_FULL_UNPADDED:
        // Circular convolution
        // We copy the first [0:h_dst-1 ; 0:w_dst-1] real part elements of out_src
        for(int i = 0 ; i < ws.h_dst ; ++i)
	  for(int j = 0 ; j < ws.w_dst ; ++j)
	    ws.dst[i*ws.w_dst+j] = ws.fft_copy->data[i*2*ws.w_fft+2*j];
        break;
      default:
        printf("Unrecognized convolution mode, possible modes are :\n");
        printf("   - LINEAR_FULL \n");
        printf("   - LINEAR_SAME \n");
        printf("   - LINEAR_SAME_UNPADDED\n");
        printf("   - LINEAR_VALID \n");
        printf("   - CIRCULAR_SAME \n");
        printf("   - CIRCULAR_SAME_PADDED \n");
        printf("   - CIRCULAR_FULL_UNPADDED\n");
        printf("   - CIRCULAR_FULL\n");
      }
  }


  

  /*   typedef enum */
  /*   { */
  /*   INVALID_MODE, // in case the workspace is not properly initialized with init_workspace(..) */
  /*   LINEAR_FULL, */
  /*   LINEAR_SAME, */
  /*   LINEAR_VALID, */
  /*   LINEAR_SAME_UNPADDED, */
  /*   CIRCULAR_SAME, */
  /*   CIRCULAR_SAME_PADDED */
  /*   } GSL_Convolution_Mode; */
    
  /*   typedef struct Workspace */
  /*   { */
  /*   gsl_fft_complex_workspace *ws_column, *ws_line; */
  /*   gsl_fft_complex_wavetable *wv_column, *wv_line; */
  /*   int h_src, w_src, h_kernel, w_kernel; */
  /*   int h_fft, w_fft; */
  /*   gsl_matrix * fft, *fft_copy; */
  /*   GSL_Convolution_Mode mode; */
  /*   double * dst; // The array containing the result */
  /*   int h_dst; // its height */
  /*   int w_dst; // its width */
  /*   } Workspace; */

  /*   void init_workspace(Workspace & ws, GSL_Convolution_Mode mode, int h_src, int w_src, int h_kernel, int w_kernel) */
  /*   { */
  /*   ws.h_src = h_src; */
  /*   ws.w_src = w_src; */
  /*   ws.h_kernel = h_kernel; */
  /*   ws.w_kernel = w_kernel; */
  /*   ws.mode = mode; */

  /*   switch(mode) */
  /*   { */
  /*   case LINEAR_FULL: */
  /*   // Full Linear convolution */
  /*   ws.h_fft = find_closest_factor(h_src + h_kernel - 1,GSL_FACTORS); */
  /*   ws.w_fft = find_closest_factor(w_src + w_kernel - 1,GSL_FACTORS); */
  /*   ws.h_dst = h_src + h_kernel-1; */
  /*   ws.w_dst = w_src + w_kernel-1; */
  /*   break; */
  /*   case LINEAR_SAME: */
  /*   // Same Linear convolution */
  /*   ws.h_fft = find_closest_factor(h_src + int(h_kernel/2.0),GSL_FACTORS); */
  /*   ws.w_fft = find_closest_factor(w_src + int(w_kernel/2.0),GSL_FACTORS); */
  /*   ws.h_dst = h_src; */
  /*   ws.w_dst = w_src; */
  /*   break; */
  /*   case LINEAR_VALID: */
  /*   // Valid Linear convolution */
  /*   if(ws.h_kernel > ws.h_src || ws.w_kernel > ws.w_src) */
  /*   { */
  /*   printf("Warning : The 'valid' convolution results in an empty matrix\n"); */
  /*   ws.h_fft = 0; */
  /*   ws.w_fft = 0; */
  /*   ws.h_dst = 0; */
  /*   ws.w_dst = 0; */
  /*   } */
  /*   else */
  /*   { */
  /*   ws.h_fft = find_closest_factor(h_src, GSL_FACTORS); */
  /*   ws.w_fft = find_closest_factor(w_src, GSL_FACTORS); */
  /*   ws.h_dst = h_src - h_kernel+1; */
  /*   ws.w_dst = w_src - w_kernel+1; */
  /*   } */
  /*   break;	 */
  /*   case LINEAR_SAME_UNPADDED: */
  /*   // Linear convolution with optimal sizes */
  /*   ws.h_fft = find_closest_factor(h_src + int(h_kernel/2.0),GSL_FACTORS); */
  /*   ws.w_fft = find_closest_factor(w_src + int(w_kernel/2.0),GSL_FACTORS); */
  /*   ws.h_dst = h_src; */
  /*   ws.w_dst = w_src; */
  /*   break; */
  /*   case CIRCULAR_SAME: */
  /*   // Circular convolution */
  /*   ws.h_fft = h_src; */
  /*   ws.w_fft = w_src; */
  /*   ws.h_dst = h_src; */
  /*   ws.w_dst = w_src; */
  /*   break; */
  /*   case CIRCULAR_SAME_PADDED: */
  /*   // Cicular convolution with optimal sizes */
  /*   ws.h_fft = find_closest_factor(h_src+h_kernel, GSL_FACTORS); */
  /*   ws.w_fft = find_closest_factor(w_src+w_kernel, GSL_FACTORS); */
  /*   break; */
  /*   default: */
  /*   printf("Unrecognized convolution mode, possible modes are :\n"); */
  /*   printf("   - LINEAR_FULL \n"); */
  /*   printf("   - LINEAR_SAME \n"); */
  /*   printf("   - LINEAR_VALID \n"); */
  /*   printf("   - LINEAR_SAME_UNPADDED \n"); */
  /*   printf("   - CIRCULAR_SAME \n"); */
  /*   printf("   - CIRCULAR_SAME_PADDED\n"); */
  /*   // TODO EXCEPTION */
  /*   } */

  /*   // We need to create a matrix holding 2 times the number of coefficients of src for the real and imaginary parts */
  /*   ws.fft = gsl_matrix_alloc(ws.h_fft, 2*ws.w_fft); */
  /*   ws.fft_copy = gsl_matrix_alloc(ws.h_fft, 2*ws.w_fft); */

  /*   ws.ws_column = gsl_fft_complex_workspace_alloc(ws.h_fft); */
  /*   ws.ws_line = gsl_fft_complex_workspace_alloc(ws.w_fft); */
  /*   ws.wv_column = gsl_fft_complex_wavetable_alloc(ws.h_fft); */
  /*   ws.wv_line = gsl_fft_complex_wavetable_alloc(ws.w_fft); */

  /*   ws.dst = new double[ws.h_dst * ws.w_dst]; */
  /*   } */

  /*   void clear_workspace(Workspace & ws) */
  /*   { */
  /*   gsl_fft_complex_workspace_free(ws.ws_column); */
  /*   gsl_fft_complex_workspace_free(ws.ws_line); */
  /*   gsl_fft_complex_wavetable_free(ws.wv_column); */
  /*   gsl_fft_complex_wavetable_free(ws.wv_line); */

  /*   gsl_matrix_free(ws.fft); */
  /*   gsl_matrix_free(ws.fft_copy); */

  /*   delete[] ws.dst; */
  /*   } */

  /*   void update_workspace(Workspace & ws, GSL_Convolution_Mode mode, int h_src, int w_src, int h_kernel, int w_kernel) */
  /*   { */
  /*   clear_workspace(ws); */
  /*   init_workspace(ws, mode, h_src, w_src, h_kernel, w_kernel); */
  /*   } */

  /*   /\*********************************\/ */
  /* /\* Linear convolution with GSL   *\/ */
  /* /\*********************************\/ */

  /* void linear_convolution(Workspace &ws, gsl_matrix * src, gsl_matrix * kernel, gsl_matrix * dst) */
  /* { */
  /*   int i_src, j_src; */
  /*   gsl_matrix_set_zero(ws.fft); */
  /*   // memcpy takes care of the strides, */
  /*   for(unsigned int i = 0 ; i < ws.h_src ; ++i) */
  /*     for(unsigned int j = 0 ; j < ws.w_src ; ++j) */
  /* 	ws.fft->data[i*2*ws.w_fft + 2*j] = src->data[i*ws.w_src + j]; */

  /*   // when zero-padding, we must ensure that the center of the kernel */
  /*   // is copied on the corners of the padded image */
  /*   for(int i = 0 ; i < ws.h_kernel ; ++i) */
  /*     { */
  /* 	i_src = i - int(ws.h_kernel/2); */
  /* 	if(i_src < 0) */
  /* 	  i_src += ws.h_fft; */

  /* 	for(int j = 0 ; j < ws.w_kernel ; ++j) */
  /* 	  { */
  /* 	    j_src = j - int(ws.w_kernel/2); */
  /* 	    if(j_src < 0) */
  /* 	      j_src += ws.w_fft; */
  /* 	    ws.fft->data[i_src * 2*ws.w_fft + 2*j_src+1] = kernel->data[i*ws.w_kernel + j]; */
  /* 	  } */
  /*     } */

  /*   // We compute the 2 forward DFT at once */
  /*   for(int i = 0 ; i < ws.h_fft ; ++i) */
  /*     { */
  /* 	// Apply the FFT on the line i */
  /* 	gsl_fft_complex_forward (&ws.fft->data[i*2*ws.w_fft],1, ws.w_fft, ws.wv_line, ws.ws_line); */
  /*     } */
  /*   for(int j = 0 ; j < ws.w_fft ; ++j) */
  /*     { */
  /* 	// Apply the FFT on the column j */
  /* 	gsl_fft_complex_forward (&ws.fft->data[2*j], ws.w_fft, ws.h_fft, ws.wv_column, ws.ws_column); */
  /*     } */

  /*   // Their element-wise product : be carefull, the matrices hold complex numbers ! */
  /*   // We need a copy of fft to perform the product properly */
  /*   double re_h, im_h, re_hs, im_hs; */
  /*   for(int i = 0 ; i < ws.h_fft ; ++i) */
  /*     { */
  /* 	for(int j = 0 ; j < ws.w_fft ; ++j) */
  /* 	  { */
  /* 	    re_h = ws.fft->data[i*2*ws.w_fft + 2*j]; */
  /* 	    im_h = ws.fft->data[i*2*ws.w_fft + 2*j+1]; */
  /* 	    re_hs = ws.fft->data[((ws.h_fft-i)%ws.h_fft)*2*ws.w_fft + ((2*ws.w_fft-2*j)%(2*ws.w_fft))]; */
  /* 	    im_hs = -ws.fft->data[((ws.h_fft-i)%ws.h_fft)*2*ws.w_fft + ((2*ws.w_fft-2*j)%(2*ws.w_fft))+1]; */

  /* 	    ws.fft_copy->data[i*2*ws.w_fft+2*j] = 0.5*(re_h*im_h - re_hs*im_hs); */
  /* 	    ws.fft_copy->data[i*2*ws.w_fft+2*j+1] = -0.25*(re_h*re_h - im_h * im_h - re_hs * re_hs + im_hs * im_hs); */
  /* 	  } */
  /*     } */

  /*   // And the inverse FFT, which is done in the similar way as before */
  /*   for(int i = 0 ; i < ws.h_fft ; ++i) */
  /*     { */
  /* 	// Apply the FFT^{-1} on the line i */
  /* 	gsl_fft_complex_inverse(&ws.fft_copy->data[i*2*ws.w_fft],1, ws.w_fft, ws.wv_line, ws.ws_line); */
  /*     } */

  /*   for(int j = 0 ; j < ws.w_fft ; ++j) */
  /*     { */
  /* 	// Apply the FFT^{-1} on the column j */
  /* 	gsl_fft_complex_inverse(&ws.fft_copy->data[2*j], ws.w_fft, ws.h_fft, ws.wv_column, ws.ws_column); */
  /*     } */

  /*   // And copy only the real part of fft_src in dst */
  /*   for(int i = 0 ; i < ws.h_src; ++i) */
  /*     { */
  /* 	for(int j = 0 ; j < ws.w_src ; ++j) */
  /* 	  { */
  /* 	    dst->data[i*ws.w_src + j] = ws.fft_copy->data[i*2*ws.w_fft+2*j]; */
  /* 	  } */
  /*     } */
  /* } */

  /* /\******************************************************\/ */
  /* /\* Linear convolution with GSL with an optimal size   *\/ */
  /* /\******************************************************\/ */

  /* void linear_convolution_optimal(Workspace ws, gsl_matrix * src, gsl_matrix * kernel, gsl_matrix * dst) */
  /* { */
  /*   int i_src, j_src; */
  /*   gsl_matrix_set_zero(ws.fft); */
  /*   for(unsigned int i = 0 ; i < ws.h_src ; ++i) */
  /*     for(unsigned int j = 0 ; j < ws.w_src ; ++j) */
  /* 	ws.fft->data[i*2*ws.w_fft + 2*j] = src->data[i*ws.w_src + j]; */

  /*   // when zero-padding, we must ensure that the center of the kernel */
  /*   // is copied on the corners of the padded image */
  /*   for(int i = 0 ; i < ws.h_kernel ; ++i) */
  /*     { */
  /* 	i_src = i - int(ws.h_kernel/2); */
  /* 	if(i_src < 0) */
  /* 	  i_src += ws.h_fft; */

  /* 	for(int j = 0 ; j < ws.w_kernel ; ++j) */
  /* 	  { */
  /* 	    j_src = j - int(ws.w_kernel/2); */
  /* 	    if(j_src < 0) */
  /* 	      j_src += ws.w_fft; */
  /* 	    ws.fft->data[i_src * 2*ws.w_fft + 2*j_src+1] = kernel->data[i * ws.w_kernel + j]; */
  /* 	  } */
  /*     } */

  /*   // We compute the 2 forward DFT at once */
  /*   for(int i = 0 ; i < ws.h_fft ; ++i) */
  /*     { */
  /* 	// Apply the FFT on the line i */
  /* 	gsl_fft_complex_forward (&ws.fft->data[i*2*ws.w_fft],1, ws.w_fft, ws.wv_line, ws.ws_line); */
  /*     } */
  /*   for(int j = 0 ; j < ws.w_fft ; ++j) */
  /*     { */
  /* 	// Apply the FFT on the column j */
  /* 	gsl_fft_complex_forward (&ws.fft->data[2*j], ws.w_fft, ws.h_fft, ws.wv_column, ws.ws_column); */
  /*     } */

  /*   // Their element-wise product : be carefull, the matrices hold complex numbers ! */
  /*   // We need a copy of fft to perform the product properly */
  /*   double re_h, im_h, re_hs, im_hs; */
  /*   for(int i = 0 ; i < ws.h_fft ; ++i) */
  /*     { */
  /* 	for(int j = 0 ; j < ws.w_fft ; ++j) */
  /* 	  { */
  /* 	    re_h = ws.fft->data[i*2*ws.w_fft + 2*j]; */
  /* 	    im_h = ws.fft->data[i*2*ws.w_fft + 2*j+1]; */
  /* 	    re_hs = ws.fft->data[((ws.h_fft-i)%ws.h_fft)*2*ws.w_fft + ((2*ws.w_fft-2*j)%(2*ws.w_fft))]; */
  /* 	    im_hs = -ws.fft->data[((ws.h_fft-i)%ws.h_fft)*2*ws.w_fft + ((2*ws.w_fft-2*j)%(2*ws.w_fft))+1]; */

  /* 	    ws.fft_copy->data[i*2*ws.w_fft+2*j] = 0.5*(re_h*im_h - re_hs*im_hs); */
  /* 	    ws.fft_copy->data[i*2*ws.w_fft+2*j+1] = -0.25*(re_h*re_h - im_h * im_h - re_hs * re_hs + im_hs * im_hs); */
  /* 	  } */
  /*     } */

  /*   // And the inverse FFT, which is done in the similar way as before */
  /*   for(int i = 0 ; i < ws.h_fft ; ++i) */
  /*     { */
  /* 	// Apply the FFT^{-1} on the line i */
  /* 	gsl_fft_complex_inverse(&ws.fft_copy->data[i*2*ws.w_fft],1, ws.w_fft, ws.wv_line, ws.ws_line); */
  /*     } */

  /*   for(int j = 0 ; j < ws.w_fft ; ++j) */
  /*     { */
  /* 	// Apply the FFT^{-1} on the column j */
  /* 	gsl_fft_complex_inverse(&ws.fft_copy->data[2*j], ws.w_fft, ws.h_fft, ws.wv_column, ws.ws_column); */
  /*     } */

  /*   // And copy only the real part of fft_src in dst */
  /*   for(int i = 0 ; i < ws.h_src; ++i) */
  /*     { */
  /* 	for(int j = 0 ; j < ws.w_src ; ++j) */
  /* 	  { */
  /* 	    dst->data[i*ws.w_src + j] = ws.fft_copy->data[i*2*ws.w_fft+2*j]; */
  /* 	  } */
  /*     } */
  /* } */

  /* /\*********************************\/ */
  /* /\* Circular convolution with GSL   *\/ */
  /* /\*********************************\/ */

  /* void circular_convolution(Workspace ws, gsl_matrix * src, gsl_matrix * kernel, gsl_matrix * dst) */
  /* { */
  /*   gsl_vector_view xv, yv; */
  /*   gsl_matrix_set_zero(ws.fft); */

  /*   for(unsigned int i = 0 ; i < ws.w_src ; ++i) */
  /*     { */
  /* 	xv = gsl_matrix_subcolumn(ws.fft,2*i,0, ws.h_src); */
  /* 	yv = gsl_matrix_column(src, i); */
  /* 	gsl_vector_memcpy(&xv.vector, &yv.vector); */
  /*     } */

  /*   // when zero-padding, we must ensure that the center of the kernel */
  /*   // is copied on the corners of the padded image */
  /*   // There are four quadrants to copy */

  /*   // The columns on the left of the center are put on the extreme right of fft_kernel */
  /*   for(unsigned int i = 0 ; i < int(ws.w_kernel/2) ; ++i) */
  /*     { */
  /* 	// The top rows of kernel are put on the bottom rows of fft_kernel */
  /* 	xv = gsl_matrix_subcolumn(ws.fft,2*ws.w_fft-2*int(ws.w_kernel/2)+2*i+1,ws.h_fft - int(ws.h_kernel/2),int(ws.h_kernel/2)); */
  /* 	yv = gsl_matrix_subcolumn(kernel, i, 0, int(ws.h_kernel/2)); */
  /* 	gsl_vector_memcpy(&xv.vector,&yv.vector); */
  /* 	// The bottom rows of the kernel are put on the top rows of fft_kernel */
  /* 	xv = gsl_matrix_subcolumn(ws.fft,2*ws.w_fft-2*int(ws.w_kernel/2)+2*i+1,0,int((ws.h_kernel+1)/2)); */
  /* 	yv = gsl_matrix_subcolumn(kernel, i, int(ws.h_kernel/2), int((ws.h_kernel+1)/2) ); */
  /* 	gsl_vector_memcpy(&xv.vector,&yv.vector); */
  /*     } */
  /*   // The columns on the right of the center are put on the extreme left of fft_kernel */
  /*   for(unsigned int i = int(ws.w_kernel/2) ; i < ws.w_kernel ; ++i) */
  /*     { */
  /* 	xv = gsl_matrix_subcolumn(ws.fft,2*(i-int(ws.w_kernel/2))+1,ws.h_fft - int(ws.h_kernel/2),int(ws.h_kernel/2)); */
  /* 	yv = gsl_matrix_subcolumn(kernel, i, 0, int(ws.h_kernel/2)); */
  /* 	// The top rows of kernel are put on the bottom rows of fft_kernel */
  /* 	gsl_vector_memcpy(&xv.vector,&yv.vector); */
  /* 	// The bottom rows of the kernel are put on the top rows of fft_kernel */
  /* 	xv = gsl_matrix_subcolumn(ws.fft,2*(i-int(ws.w_kernel/2))+1,0,int((ws.h_kernel+1)/2)); */
  /* 	yv = gsl_matrix_subcolumn(kernel, i, int(ws.h_kernel/2), int((ws.h_kernel+1)/2)); */
  /* 	gsl_vector_memcpy(&xv.vector,&yv.vector); */
  /*     } */

  /*   // We compute the 2 forward DFT at once */
  /*   for(int i = 0 ; i < ws.h_fft ; ++i) */
  /*     { */
  /* 	// Apply the FFT on the line i */
  /* 	gsl_fft_complex_forward (&ws.fft->data[i*2*ws.w_fft],1, ws.w_fft, ws.wv_line, ws.ws_line); */
  /*     } */
  /*   for(int j = 0 ; j < ws.w_fft ; ++j) */
  /*     { */
  /* 	// Apply the FFT on the column j */
  /* 	gsl_fft_complex_forward (&ws.fft->data[2*j],ws.w_fft, ws.h_fft, ws.wv_column, ws.ws_column); */
  /*     } */

  /*   // Their element-wise product : be carefull, the matrices hold complex numbers ! */
  /*   // We need a copy of fft to perform the product properly */
  /*   double re_h, im_h, re_hs, im_hs; */
  /*   for(int i = 0 ; i < ws.h_fft ; ++i) */
  /*     { */
  /* 	for(int j = 0 ; j < ws.w_fft ; ++j) */
  /* 	  { */
  /* 	    re_h = gsl_matrix_get(ws.fft,i, 2*j); */
  /* 	    im_h = gsl_matrix_get(ws.fft,i, 2*j+1); */
  /* 	    re_hs = gsl_matrix_get(ws.fft,(ws.h_fft-i)%ws.h_fft, ((2*ws.w_fft-2*j)%(2*ws.w_fft))); */
  /* 	    im_hs = - gsl_matrix_get(ws.fft,(ws.h_fft-i)%ws.h_fft, ((2*ws.w_fft-2*j)%(2*ws.w_fft))+1); */

  /* 	    gsl_matrix_set(ws.fft_copy, i, 2*j, 0.5*(re_h*im_h - re_hs*im_hs)); */
  /* 	    gsl_matrix_set(ws.fft_copy, i, 2*j+1, -0.25*(re_h*re_h - im_h * im_h - re_hs * re_hs + im_hs * im_hs)); */
  /* 	  } */
  /*     } */

  /*   // And the inverse FFT, which is done in the similar way as before */
  /*   for(int i = 0 ; i < ws.h_fft ; ++i) */
  /*     { */
  /* 	// Apply the FFT^{-1} on the line i */
  /* 	gsl_fft_complex_inverse(&ws.fft_copy->data[i*2*ws.w_fft],1, ws.w_fft, ws.wv_line, ws.ws_line); */
  /*     } */

  /*   for(int j = 0 ; j < ws.w_fft ; ++j) */
  /*     { */
  /* 	// Apply the FFT^{-1} on the column j */
  /* 	gsl_fft_complex_inverse(&ws.fft_copy->data[2*j],ws.w_fft, ws.h_fft, ws.wv_column, ws.ws_column); */
  /*     } */

  /*   // And copy only the real part of fft_src in dst */
  /*   for(unsigned int i = 0 ; i < ws.w_src ; ++i) */
  /*     { */
  /* 	xv = gsl_matrix_subcolumn(ws.fft_copy, 2*i,0,ws.h_src); */
  /* 	gsl_matrix_set_col(dst, i, &xv.vector); */
  /*     } */
  /* } */

  /* /\******************************************************\/ */
  /* /\* Circular convolution with GSL with an optimal size   *\/ */
  /* /\******************************************************\/ */

  /* void circular_convolution_optimal(Workspace ws, gsl_matrix * src, gsl_matrix * kernel, gsl_matrix * dst) */
  /* { */
  /*   // In order to optimze the size for a circular convolution */
  /*   // it is sufficient to remind that a circular convolution is the restriction to the central part */
  /*   // of a linear convolution performed on a larger image with wrapped around values on the borders */


  /*   // Copy and wrap around src */
  /*   // fft is filled differently for 10 regions (where the regions filled in with 0 is counted as a single region) */
  /*   // wrap bottom right | wrap bottom | wrap bottom left | 0 */
  /*   //     wrap right    |     src     |   wrap left      | 0 */
  /*   // wrap top  right   |  wrap top   |  wrap top left   | 0 */
  /*   //        0          |      0      |       0          | 0 */

  /*   gsl_matrix_set_zero(ws.fft); */
  /*   int i_src, j_src; */
  /*   for(int i = 0 ; i < ws.h_src + ws.h_kernel ; ++i) */
  /*     { */
  /* 	i_src = i - int((ws.h_kernel+1)/2); */
  /* 	if(i_src < 0) */
  /* 	  i_src += ws.h_src; */
  /* 	else if(i_src >= ws.h_src) */
  /* 	  i_src -= ws.h_src; */
  /* 	for(int j = 0 ; j < ws.w_src + ws.w_kernel ; ++j) */
  /* 	  { */
  /* 	    j_src = j - int((ws.w_kernel+1)/2); */
  /* 	    if(j_src < 0) */
  /* 	      j_src += ws.w_src; */
  /* 	    else if(j_src >= ws.w_src) */
  /* 	      j_src -= ws.w_src; */

  /* 	    ws.fft->data[i * 2*ws.w_fft + 2*j] = src->data[i_src * ws.w_src + j_src]; */
  /* 	  } */
  /*     } */

  /*   ////// */
  /*   // Given this new source image, the following is exactly the same as for performing a linear convolution in GSL */
  /*   ////// */

  /*   // when zero-padding, we must ensure that the center of the kernel */
  /*   // is copied on the corners of the padded image */
  /*   for(int i = 0 ; i < ws.h_kernel ; ++i) */
  /*     { */
  /* 	i_src = i - int(ws.h_kernel/2); */
  /* 	if(i_src < 0) */
  /* 	  i_src += ws.h_fft; */

  /* 	for(int j = 0 ; j < ws.w_kernel ; ++j) */
  /* 	  { */
  /* 	    j_src = j - int(ws.w_kernel/2); */
  /* 	    if(j_src < 0) */
  /* 	      j_src += ws.w_fft; */
  /* 	    ws.fft->data[i_src * 2*ws.w_fft + 2*j_src+1] = kernel->data[i * ws.w_kernel + j]; */
  /* 	  } */
  /*     } */

  /*   // We compute the 2 forward DFT at once */
  /*   for(int i = 0 ; i < ws.h_fft ; ++i) */
  /*     { */
  /* 	// Apply the FFT on the line i */
  /* 	gsl_fft_complex_forward (&ws.fft->data[i*2*ws.w_fft],1, ws.w_fft, ws.wv_line, ws.ws_line); */
  /*     } */
  /*   for(int j = 0 ; j < ws.w_fft ; ++j) */
  /*     { */
  /* 	// Apply the FFT on the column j */
  /* 	gsl_fft_complex_forward (&ws.fft->data[2*j],ws.w_fft, ws.h_fft, ws.wv_column, ws.ws_column); */
  /*     } */

  /*   // Their element-wise product : be carefull, the matrices hold complex numbers ! */
  /*   // We need a copy of fft to perform the product properly */
  /*   double re_h, im_h, re_hs, im_hs; */
  /*   for(int i = 0 ; i < ws.h_fft ; ++i) */
  /*     { */
  /* 	for(int j = 0 ; j < ws.w_fft ; ++j) */
  /* 	  { */
  /* 	    re_h = ws.fft->data[i*2*ws.w_fft + 2*j]; */
  /* 	    im_h = ws.fft->data[i*2*ws.w_fft + 2*j+1]; */
  /* 	    re_hs = ws.fft->data[((ws.h_fft-i)%ws.h_fft)*2*ws.w_fft + ((2*ws.w_fft-2*j)%(2*ws.w_fft))]; */
  /* 	    im_hs = -ws.fft->data[((ws.h_fft-i)%ws.h_fft)*2*ws.w_fft + ((2*ws.w_fft-2*j)%(2*ws.w_fft))+1]; */

  /* 	    ws.fft_copy->data[i*2*ws.w_fft+2*j] = 0.5*(re_h*im_h - re_hs*im_hs); */
  /* 	    ws.fft_copy->data[i*2*ws.w_fft+2*j+1] = -0.25*(re_h*re_h - im_h * im_h - re_hs * re_hs + im_hs * im_hs); */
  /* 	  } */
  /*     } */

  /*   // And the inverse FFT, which is done in the similar way as before */
  /*   for(int i = 0 ; i < ws.h_fft ; ++i) */
  /*     { */
  /* 	// Apply the FFT^{-1} on the line i */
  /* 	gsl_fft_complex_inverse(&ws.fft_copy->data[i*2*ws.w_fft],1, ws.w_fft, ws.wv_line, ws.ws_line); */
  /*     } */

  /*   for(int j = 0 ; j < ws.w_fft ; ++j) */
  /*     { */
  /* 	// Apply the FFT^{-1} on the column j */
  /* 	gsl_fft_complex_inverse(&ws.fft_copy->data[2*j],ws.w_fft, ws.h_fft, ws.wv_column, ws.ws_column); */
  /*     } */

  /*   // And copy only the real part of the central part of fft_src in dst */
  /*   for(int i = 0 ; i < ws.h_src; ++i) */
  /*     for(int j = 0 ; j < ws.w_src ; ++j) */
  /* 	dst->data[i*ws.w_src + j] = ws.fft_copy->data[(i+int((ws.h_kernel+1)/2))*2*ws.w_fft+2*(int((ws.w_kernel+1)/2)+j)]; */
  /* } */

  /* void convolve(Workspace &ws, double * src, double * kernel, double * dst) */
  /* { */
  /*   /\* */
  /*     gsl_matrix_view src_view = gsl_matrix_view_array(src, ws.h_src, ws.w_src); */
  /*     gsl_matrix_view kernel_view = gsl_matrix_view_array(kernel, ws.h_kernel, ws.w_kernel); */
  /*     gsl_matrix_view dst_view = gsl_matrix_view_array(dst, ws.h_src, ws.w_src); */
    
  /*     switch(ws.mode) */
  /*     { */
  /*     case LINEAR: */
  /*     linear_convolution(ws, &src_view.matrix, &kernel_view.matrix, &dst_view.matrix); */
  /*     break; */
  /*     case LINEAR_OPTIMAL: */
  /*     linear_convolution_optimal(ws, &src_view.matrix, &kernel_view.matrix, &dst_view.matrix); */
  /*     break; */
  /*     case CIRCULAR: */
  /*     circular_convolution(ws, &src_view.matrix, &kernel_view.matrix, &dst_view.matrix); */
  /*     break; */
  /*     case CIRCULAR_OPTIMAL: */
  /*     circular_convolution_optimal(ws, &src_view.matrix, &kernel_view.matrix, &dst_view.matrix); */
  /*     break; */
  /*     default: */
  /*     printf("Unrecognized convolution mode, possible modes are :\n"); */
  /*     printf("   - LINEAR \n"); */
  /*     printf("   - LINEAR_OPTIMAL \n"); */
  /*     printf("   - CIRCULAR \n"); */
  /*     printf("   - CIRCULAR_OPTIMAL\n"); */
  /*     // TODO EXCEPTION */
  /*     } */
  /*   *\/ */

  /*   // First clean up in_src; */
  /*   for(int i = 0 ; i < ws.h_fft ; ++i) */
  /*     { */
  /* 	for(int j = 0 ; j < ws.w_fft ; ++j) */
  /* 	  { */
  /* 	    ws.in_src[i*ws.w_fft+j][0] = 0.0; */
  /* 	    ws.in_src[i*ws.w_fft+j][1] = 0.0; */
  /* 	  } */
  /*     } */









  /*   // We compute the 2 forward DFT at once */
  /*   for(int i = 0 ; i < ws.h_fft ; ++i) */
  /*     { */
  /* 	// Apply the FFT on the line i */
  /* 	gsl_fft_complex_forward (&ws.fft->data[i*2*ws.w_fft],1, ws.w_fft, ws.wv_line, ws.ws_line); */
  /*     } */
  /*   for(int j = 0 ; j < ws.w_fft ; ++j) */
  /*     { */
  /* 	// Apply the FFT on the column j */
  /* 	gsl_fft_complex_forward (&ws.fft->data[2*j],ws.w_fft, ws.h_fft, ws.wv_column, ws.ws_column); */
  /*     } */

  /*   // Their element-wise product : be carefull, the matrices hold complex numbers ! */
  /*   // We need a copy of fft to perform the product properly */
  /*   double re_h, im_h, re_hs, im_hs; */
  /*   for(int i = 0 ; i < ws.h_fft ; ++i) */
  /*     { */
  /* 	for(int j = 0 ; j < ws.w_fft ; ++j) */
  /* 	  { */
  /* 	    re_h = gsl_matrix_get(ws.fft,i, 2*j); */
  /* 	    im_h = gsl_matrix_get(ws.fft,i, 2*j+1); */
  /* 	    re_hs = gsl_matrix_get(ws.fft,(ws.h_fft-i)%ws.h_fft, ((2*ws.w_fft-2*j)%(2*ws.w_fft))); */
  /* 	    im_hs = - gsl_matrix_get(ws.fft,(ws.h_fft-i)%ws.h_fft, ((2*ws.w_fft-2*j)%(2*ws.w_fft))+1); */

  /* 	    gsl_matrix_set(ws.fft_copy, i, 2*j, 0.5*(re_h*im_h - re_hs*im_hs)); */
  /* 	    gsl_matrix_set(ws.fft_copy, i, 2*j+1, -0.25*(re_h*re_h - im_h * im_h - re_hs * re_hs + im_hs * im_hs)); */
  /* 	  } */
  /*     } */

  /*   // And the inverse FFT, which is done in the similar way as before */
  /*   for(int i = 0 ; i < ws.h_fft ; ++i) */
  /*     { */
  /* 	// Apply the FFT^{-1} on the line i */
  /* 	gsl_fft_complex_inverse(&ws.fft_copy->data[i*2*ws.w_fft],1, ws.w_fft, ws.wv_line, ws.ws_line); */
  /*     } */

  /*   for(int j = 0 ; j < ws.w_fft ; ++j) */
  /*     { */
  /* 	// Apply the FFT^{-1} on the column j */
  /* 	gsl_fft_complex_inverse(&ws.fft_copy->data[2*j],ws.w_fft, ws.h_fft, ws.wv_column, ws.ws_column); */
  /*     } */
  /* } */
  

    }


#endif
