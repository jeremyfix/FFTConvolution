#ifndef CONVOLUTION_FFTW_H
#define CONVOLUTION_FFTW_H

#include <cassert>
#include <fftw3.h>
#include "factorize.h"

namespace FFTW_Convolution 
{

  int FFTW_FACTORS[7] = {13,11,7,5,3,2,0}; // end with zero to detect the end of the array

  typedef enum
  {
    LINEAR,
    LINEAR_OPTIMAL,
    CIRCULAR,
    CIRCULAR_OPTIMAL
  } FFTW_Convolution_Mode;

  typedef struct Workspace
  {
    fftw_complex * in_src, *out_src;
    int h_src, w_src, h_kernel, w_kernel;
    int w_fftw, h_fftw;
    fftw_plan p_forw;
    fftw_plan p_back;
    FFTW_Convolution_Mode mode;
  } Workspace;


  void init_workspace(Workspace & ws, FFTW_Convolution_Mode mode, int h_src, int w_src, int h_kernel, int w_kernel)
  {
    ws.h_src = h_src;
    ws.w_src = w_src;
    ws.h_kernel = h_kernel;
    ws.w_kernel = w_kernel;
    ws.mode = mode;

    switch(mode)
      {
      case LINEAR:
	// Linear convolution
	ws.h_fftw = h_src + int((h_kernel+1)/2);
	ws.w_fftw = w_src + int((w_kernel+1)/2);
	break;
      case LINEAR_OPTIMAL:
	// Linear convolution with optimal sizes
	ws.h_fftw = find_closest_factor(h_src + int((h_kernel+1)/2),FFTW_FACTORS);
	ws.w_fftw = find_closest_factor(w_src + int((w_kernel+1)/2), FFTW_FACTORS);
	break;
      case CIRCULAR:
	// Circular convolution
	assert(h_kernel <= h_src && w_kernel <= w_src);
	ws.h_fftw = h_src;
	ws.w_fftw = w_src;
	break;
      case CIRCULAR_OPTIMAL:
	// Cicular convolution with optimal sizes
	assert(h_kernel <= h_src && w_kernel <= w_src);
	ws.h_fftw = find_closest_factor(h_src+h_kernel, FFTW_FACTORS);
	ws.w_fftw = find_closest_factor(w_src+w_kernel, FFTW_FACTORS);
	break;
      default:
	printf("Unrecognized convolution mode, possible modes are :\n");
	printf("   - FFTW_LINEAR \n");
	printf("   - FFTW_LINEAR_OPTIMAL \n");
	printf("   - FFTW_CIRCULAR \n");
	printf("   - FFTW_CIRCULAR_OPTIMAL\n");
	// TODO EXCEPTION
      }
    ws.in_src = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * ws.h_fftw * ws.w_fftw);
    ws.out_src = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * ws.h_fftw * ws.w_fftw);
    ws.p_forw = fftw_plan_dft_2d(ws.h_fftw, ws.w_fftw, ws.in_src, ws.out_src, FFTW_FORWARD, FFTW_ESTIMATE);
    ws.p_back = fftw_plan_dft_2d(ws.h_fftw, ws.w_fftw, ws.in_src, ws.out_src, FFTW_BACKWARD, FFTW_ESTIMATE);
  }

  void clear_workspace(Workspace & ws)
  {
    fftw_destroy_plan(ws.p_forw);
    fftw_destroy_plan(ws.p_back);
    fftw_free(ws.in_src);
    fftw_free(ws.out_src);
  }

  void update_workspace(Workspace & ws, FFTW_Convolution_Mode mode, int h_src, int w_src, int h_kernel, int w_kernel)
  {
    clear_workspace(ws);
    init_workspace(ws, mode, h_src, w_src, h_kernel, w_kernel);
  }

  /* ******************************************************************************************************** */
  /*        Linear convolution with FFTW3, computing 2DFT at the same time                                    */
  /* ******************************************************************************************************** */

  void linear_convolution(Workspace &ws, double * src, double * kernel, double * dst)
  {
    // We need to fill the real part of in_src with the src image
    for(int i = 0 ; i < ws.h_fftw ; ++i)
      {
	for(int j = 0 ; j < ws.w_fftw ; ++j)
	  {
	    if( i < ws.h_src && j < ws.w_src)
	      ws.in_src[i * ws.w_fftw + j][0] = src[i*ws.w_src+j];
	    else
	      ws.in_src[i * ws.w_fftw + j][0] = 0.0;
	    ws.in_src[i * ws.w_fftw + j][1] = 0.0;
	  }
      }

    // We padd and wrap the kernel so that it has the same size as the src image, and that the center of the
    // filter is in (0,0)
    int i_src, j_src;
    for(int i = 0 ; i < ws.h_kernel ; ++i)
      {
	i_src = i - int(ws.h_kernel/2);
	if(i_src < 0)
	  i_src += ws.h_fftw;

	for(int j = 0 ; j < ws.w_kernel ; ++j)
	  {
	    j_src = j - int(ws.w_kernel/2);
	    if(j_src < 0)
	      j_src += ws.w_fftw;
	    ws.in_src[i_src * ws.w_fftw + j_src][1] = kernel[i * ws.w_kernel + j];
	  }
      }

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

    // Now we just need to copy the right part of in_src into dst
    for(int i  = 0 ; i < ws.h_src ; ++i)
      {
	for(int j = 0 ; j < ws.w_src ; ++j)
	  {
	    dst[i*ws.w_src+ j] = ws.out_src[i * ws.w_fftw + j][0]/ double(ws.w_fftw * ws.h_fftw);
	  }
      }
  }

  /******************************************************/
  /*  Linear convolution with FFTW with an optimal size */
  /******************************************************/

  void linear_convolution_optimal(Workspace &ws, double * src, double * kernel, double * dst)
  {
    // We need to fill the real part of in_src with the src image
    for(int i = 0 ; i < ws.h_fftw ; ++i)
      {
	for(int j = 0 ; j < ws.w_fftw ; ++j)
	  {
	    if( i < ws.h_src && j < ws.w_src)
	      ws.in_src[i * ws.w_fftw + j][0] = src[i*ws.w_src+j];
	    else
	      ws.in_src[i * ws.w_fftw + j][0] = 0.0;
	    ws.in_src[i * ws.w_fftw + j][1] = 0.0;
	  }
      }

    // We padd and wrap the kernel so that it has the same size as the src image, and that the center of the
    // filter is in (0,0)
    int i_src, j_src;
    for(int i = 0 ; i < ws.h_kernel ; ++i)
      {
	i_src = i - int(ws.h_kernel/2);
	if(i_src < 0)
	  i_src += ws.h_fftw;

	for(int j = 0 ; j < ws.w_kernel ; ++j)
	  {
	    j_src = j - int(ws.w_kernel/2);
	    if(j_src < 0)
	      j_src += ws.w_fftw;
	    ws.in_src[i_src * ws.w_fftw + j_src][1] = kernel[i * ws.w_kernel + j];
	  }
      }

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

    // Now we just need to copy the right part of in_src into dst
    for(int i  = 0 ; i < ws.h_src ; ++i)
      {
	for(int j = 0 ; j < ws.w_src ; ++j)
	  {
	    dst[i*ws.w_src+ j] = ws.out_src[i * ws.w_fftw + j][0]/ double(ws.w_fftw * ws.h_fftw);
	  }
      }
  }

  /************************************/
  /* Circular convolution with FFTW   */
  /************************************/

  void circular_convolution(Workspace &ws, double * src, double * kernel, double * dst)
  {
    // We need to fill the real part of in_src with the src image
    for(int i = 0 ; i < ws.h_fftw ; ++i)
      {
	for(int j = 0 ; j < ws.w_fftw ; ++j)
	  {
	    if( i < ws.h_src && j < ws.w_src)
	      ws.in_src[i * ws.w_fftw + j][0] = src[i*ws.w_src+j];
	    else
	      ws.in_src[i * ws.w_fftw + j][0] = 0.0;
	    ws.in_src[i * ws.w_fftw + j][1] = 0.0;
	  }
      }

    // We padd and wrap the kernel so that it has the same size as the src image, and that the center of the
    // filter is in (0,0)
    int i_src, j_src;
    for(int i = 0 ; i < ws.h_kernel ; ++i)
      {
	i_src = i - int(ws.h_kernel/2);
	if(i_src < 0)
	  i_src += ws.h_fftw;

	for(int j = 0 ; j < ws.w_kernel ; ++j)
	  {
	    j_src = j - int(ws.w_kernel/2);
	    if(j_src < 0)
	      j_src += ws.w_fftw;
	    ws.in_src[i_src * ws.w_fftw + j_src][1] = kernel[i * ws.w_kernel + j];
	  }
      }

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

    // Now we just need to copy the right part of in_src into dst
    for(int i  = 0 ; i < ws.h_src ; ++i)
      {
	for(int j = 0 ; j < ws.w_src ; ++j)
	  {
	    dst[i*ws.w_src+ j] = ws.out_src[i * ws.w_fftw + j][0]/ double(ws.w_fftw * ws.h_fftw);
	  }
      }
  }

  /******************************************************/
  /* Circular convolution with FFTW with an optimal size   */
  /******************************************************/

  void circular_convolution_optimal(Workspace &ws,double * src, double * kernel, double * dst)
  {
    for(int i = 0 ; i < ws.h_fftw ; ++i)
      {
	for(int j = 0 ; j < ws.w_fftw ; ++j)
	  {
	    ws.in_src[i*ws.w_fftw + j][0] = 0.0;
	    ws.in_src[i*ws.w_fftw + j][1] = 0.0;
	  }
      }

    // Copy and wrap around src
    // fft is filled differently for 10 regions (where the regions filled in with 0 is counted as a single region)
    //
    // wrap bottom right | wrap bottom | wrap bottom left | 0
    //     wrap right    |     src     |   wrap left      | 0
    // wrap top  right   |  wrap top   |  wrap top left   | 0
    //        0          |      0      |       0          | 0

    int i_src, j_src;
    for(int i = 0 ; i < ws.h_src + ws.h_kernel ; ++i)
      {
	i_src = i - int((ws.h_kernel+1)/2);
	if(i_src < 0)
	  i_src += ws.h_src;
	else if(i_src >= ws.h_src)
	  i_src -= ws.h_src;
	for(int j = 0 ; j < ws.w_src + ws.w_kernel ; ++j)
	  {
	    j_src = j - int((ws.w_kernel+1)/2);
	    if(j_src < 0)
	      j_src += ws.w_src;
	    else if(j_src >= ws.w_src)
	      j_src -= ws.w_src;

	    ws.in_src[i * ws.w_fftw + j][0] = src[i_src * ws.w_src + j_src];
	  }
      }

    //////
    // Given this new source image, the following is exactly the same as for performing a linear convolution in GSL
    //////

    // We padd and wrap the kernel so that it has the same size as the src image, and that the center of the
    // filter is in (0,0)
    for(int i = 0 ; i < ws.h_kernel ; ++i)
      {
	i_src = i - int(ws.h_kernel/2);
	if(i_src < 0)
	  i_src += ws.h_fftw;

	for(int j = 0 ; j < ws.w_kernel ; ++j)
	  {
	    j_src = j - int(ws.w_kernel/2);
	    if(j_src < 0)
	      j_src += ws.w_fftw;
	    ws.in_src[i_src * ws.w_fftw + j_src][1] = kernel[i * ws.w_kernel + j];
	  }
      }

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

    // Now we just need to copy the right part of in_src into dst
    for(int i  = 0 ; i < ws.h_src ; ++i)
      for(int j = 0 ; j < ws.w_src ; ++j)
	dst[i*ws.w_src+ j] = ws.out_src[(i+int((ws.h_kernel+1)/2)) * ws.w_fftw + j + int((ws.w_kernel+1)/2)][0]/ double(ws.w_fftw * ws.h_fftw);
  }

  void convolve(Workspace &ws, double * src, double * kernel, double * dst)
  {
    switch(ws.mode)
      {
      case LINEAR:
	linear_convolution(ws, src, kernel, dst);
	break;
      case LINEAR_OPTIMAL:
	linear_convolution_optimal(ws, src, kernel, dst);
	break;
      case CIRCULAR:
	circular_convolution(ws, src, kernel, dst);
	break;
      case CIRCULAR_OPTIMAL:
	circular_convolution_optimal(ws, src, kernel, dst);
	break;
      default:
	printf("Unrecognized convolution mode, possible modes are :\n");
	printf("   - FFTW_LINEAR \n");
	printf("   - FFTW_LINEAR_OPTIMAL \n");
	printf("   - FFTW_CIRCULAR \n");
	printf("   - FFTW_CIRCULAR_OPTIMAL\n");
	// TODO EXCEPTION
      }
  }

}

#endif
