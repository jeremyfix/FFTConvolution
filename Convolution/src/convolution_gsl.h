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
    LINEAR,
    LINEAR_OPTIMAL,
    CIRCULAR,
    CIRCULAR_OPTIMAL
  } GSL_Convolution_Mode;
    
  typedef struct Workspace
  {
    gsl_fft_complex_workspace *ws_column, *ws_line;
    gsl_fft_complex_wavetable *wv_column, *wv_line;
    int h_src, w_src, h_kernel, w_kernel;
    int h_res, w_res;
    gsl_matrix * fft, *fft_copy;
    GSL_Convolution_Mode mode;
  } Workspace;

  void init_workspace(Workspace & ws, GSL_Convolution_Mode mode, int h_src, int w_src, int h_kernel, int w_kernel)
  {
    ws.h_src = h_src;
    ws.w_src = w_src;
    ws.h_kernel = h_kernel;
    ws.w_kernel = w_kernel;
    ws.mode = mode;

    switch(mode)
      {
      case LINEAR:
	ws.h_res = h_src + int((h_kernel+1)/2);
	ws.w_res = w_src + int((w_kernel+1)/2);
	// We need to create a matrix holding 2 times the number of coefficients of src for the real and imaginary parts
	ws.fft = gsl_matrix_alloc(ws.h_res, 2*ws.w_res);
	ws.fft_copy = gsl_matrix_alloc(ws.h_res, 2*ws.w_res);
	break;
      case LINEAR_OPTIMAL:
	ws.h_res = find_closest_factor(h_src + int(h_kernel+1)/2, GSL_FACTORS);
	ws.w_res = find_closest_factor(w_src + int(w_kernel+1)/2, GSL_FACTORS);
	// We need to create a matrix holding 2 times the number of coefficients of src for the real and imaginary parts
	ws.fft = gsl_matrix_alloc(ws.h_res, 2*ws.w_res);
	ws.fft_copy = gsl_matrix_alloc(ws.h_res, 2*ws.w_res);
	break;
      case CIRCULAR:
	break;
      case CIRCULAR_OPTIMAL:
	break;
      }
    ws.ws_column = gsl_fft_complex_workspace_alloc(ws.h_res);
    ws.ws_line = gsl_fft_complex_workspace_alloc(ws.w_res);
    ws.wv_column = gsl_fft_complex_wavetable_alloc(ws.h_res);
    ws.wv_line = gsl_fft_complex_wavetable_alloc(ws.w_res);

  }

  void clear_workspace(Workspace & ws)
  {
    gsl_fft_complex_workspace_free(ws.ws_column);
    gsl_fft_complex_workspace_free(ws.ws_line);
    gsl_fft_complex_wavetable_free(ws.wv_column);
    gsl_fft_complex_wavetable_free(ws.wv_line);

    gsl_matrix_free(ws.fft);
    gsl_matrix_free(ws.fft_copy);
  }

  void update_workspace(Workspace & ws, GSL_Convolution_Mode mode, int h_src, int w_src, int h_kernel, int w_kernel)
  {
    clear_workspace(ws);
    init_workspace(ws, mode, h_src, w_src, h_kernel, w_kernel);
  }

  /*********************************/
  /* Linear convolution with GSL   */
  /*********************************/

  void linear_convolution(Workspace &ws, gsl_matrix * src, gsl_matrix * kernel, gsl_matrix * dst)
  {
    int i_src, j_src;
    gsl_matrix_set_zero(ws.fft);
    // memcpy takes care of the strides,
    for(unsigned int i = 0 ; i < ws.h_src ; ++i)
      for(unsigned int j = 0 ; j < ws.w_src ; ++j)
	ws.fft->data[i*2*ws.w_res + 2*j] = src->data[i*ws.w_src + j];

    // when zero-padding, we must ensure that the center of the kernel
    // is copied on the corners of the padded image
    for(int i = 0 ; i < ws.h_kernel ; ++i)
      {
	i_src = i - int(ws.h_kernel/2);
	if(i_src < 0)
	  i_src += ws.h_res;

	for(int j = 0 ; j < ws.w_kernel ; ++j)
	  {
	    j_src = j - int(ws.w_kernel/2);
	    if(j_src < 0)
	      j_src += ws.w_res;
	    ws.fft->data[i_src * 2*ws.w_res + 2*j_src+1] = kernel->data[i*ws.w_kernel + j];
	  }
      }

    // We compute the 2 forward DFT at once
    for(int i = 0 ; i < ws.h_res ; ++i)
      {
	// Apply the FFT on the line i
	gsl_fft_complex_forward (&ws.fft->data[i*2*ws.w_res],1, ws.w_res, ws.wv_line, ws.ws_line);
      }
    for(int j = 0 ; j < ws.w_res ; ++j)
      {
	// Apply the FFT on the column j
	gsl_fft_complex_forward (&ws.fft->data[2*j], ws.w_res, ws.h_res, ws.wv_column, ws.ws_column);
      }

    // Their element-wise product : be carefull, the matrices hold complex numbers !
    // We need a copy of fft to perform the product properly
    double re_h, im_h, re_hs, im_hs;
    for(int i = 0 ; i < ws.h_res ; ++i)
      {
	for(int j = 0 ; j < ws.w_res ; ++j)
	  {
	    re_h = ws.fft->data[i*2*ws.w_res + 2*j];
	    im_h = ws.fft->data[i*2*ws.w_res + 2*j+1];
	    re_hs = ws.fft->data[((ws.h_res-i)%ws.h_res)*2*ws.w_res + ((2*ws.w_res-2*j)%(2*ws.w_res))];
	    im_hs = -ws.fft->data[((ws.h_res-i)%ws.h_res)*2*ws.w_res + ((2*ws.w_res-2*j)%(2*ws.w_res))+1];

	    ws.fft_copy->data[i*2*ws.w_res+2*j] = 0.5*(re_h*im_h - re_hs*im_hs);
	    ws.fft_copy->data[i*2*ws.w_res+2*j+1] = -0.25*(re_h*re_h - im_h * im_h - re_hs * re_hs + im_hs * im_hs);
	  }
      }

    // And the inverse FFT, which is done in the similar way as before
    for(int i = 0 ; i < ws.h_res ; ++i)
      {
	// Apply the FFT^{-1} on the line i
	gsl_fft_complex_inverse(&ws.fft_copy->data[i*2*ws.w_res],1, ws.w_res, ws.wv_line, ws.ws_line);
      }

    for(int j = 0 ; j < ws.w_res ; ++j)
      {
	// Apply the FFT^{-1} on the column j
	gsl_fft_complex_inverse(&ws.fft_copy->data[2*j], ws.w_res, ws.h_res, ws.wv_column, ws.ws_column);
      }

    // And copy only the real part of fft_src in dst
    for(int i = 0 ; i < ws.h_src; ++i)
      {
	for(int j = 0 ; j < ws.w_src ; ++j)
	  {
	    dst->data[i*ws.w_src + j] = ws.fft_copy->data[i*2*ws.w_res+2*j];
	  }
      }
  }

  /******************************************************/
  /* Linear convolution with GSL with an optimal size   */
  /******************************************************/

  void linear_convolution_optimal(Workspace ws, gsl_matrix * src, gsl_matrix * kernel, gsl_matrix * dst)
  {
    int i_src, j_src;
    gsl_matrix_set_zero(ws.fft);
    for(unsigned int i = 0 ; i < ws.h_src ; ++i)
      for(unsigned int j = 0 ; j < ws.w_src ; ++j)
	ws.fft->data[i*2*ws.w_res + 2*j] = src->data[i*ws.w_src + j];

    // when zero-padding, we must ensure that the center of the kernel
    // is copied on the corners of the padded image
    for(int i = 0 ; i < ws.h_kernel ; ++i)
      {
	i_src = i - int(ws.h_kernel/2);
	if(i_src < 0)
	  i_src += ws.h_res;

	for(int j = 0 ; j < ws.w_kernel ; ++j)
	  {
	    j_src = j - int(ws.w_kernel/2);
	    if(j_src < 0)
	      j_src += ws.w_res;
	    ws.fft->data[i_src * 2*ws.w_res + 2*j_src+1] = kernel->data[i * ws.w_kernel + j];
	  }
      }

    // We compute the 2 forward DFT at once
    for(int i = 0 ; i < ws.h_res ; ++i)
      {
	// Apply the FFT on the line i
	gsl_fft_complex_forward (&ws.fft->data[i*2*ws.w_res],1, ws.w_res, ws.wv_line, ws.ws_line);
      }
    for(int j = 0 ; j < ws.w_res ; ++j)
      {
	// Apply the FFT on the column j
	gsl_fft_complex_forward (&ws.fft->data[2*j], ws.w_res, ws.h_res, ws.wv_column, ws.ws_column);
      }

    // Their element-wise product : be carefull, the matrices hold complex numbers !
    // We need a copy of fft to perform the product properly
    double re_h, im_h, re_hs, im_hs;
    for(int i = 0 ; i < ws.h_res ; ++i)
      {
	for(int j = 0 ; j < ws.w_res ; ++j)
	  {
	    re_h = ws.fft->data[i*2*ws.w_res + 2*j];
	    im_h = ws.fft->data[i*2*ws.w_res + 2*j+1];
	    re_hs = ws.fft->data[((ws.h_res-i)%ws.h_res)*2*ws.w_res + ((2*ws.w_res-2*j)%(2*ws.w_res))];
	    im_hs = -ws.fft->data[((ws.h_res-i)%ws.h_res)*2*ws.w_res + ((2*ws.w_res-2*j)%(2*ws.w_res))+1];

	    ws.fft_copy->data[i*2*ws.w_res+2*j] = 0.5*(re_h*im_h - re_hs*im_hs);
	    ws.fft_copy->data[i*2*ws.w_res+2*j+1] = -0.25*(re_h*re_h - im_h * im_h - re_hs * re_hs + im_hs * im_hs);
	  }
      }

    // And the inverse FFT, which is done in the similar way as before
    for(int i = 0 ; i < ws.h_res ; ++i)
      {
	// Apply the FFT^{-1} on the line i
	gsl_fft_complex_inverse(&ws.fft_copy->data[i*2*ws.w_res],1, ws.w_res, ws.wv_line, ws.ws_line);
      }

    for(int j = 0 ; j < ws.w_res ; ++j)
      {
	// Apply the FFT^{-1} on the column j
	gsl_fft_complex_inverse(&ws.fft_copy->data[2*j], ws.w_res, ws.h_res, ws.wv_column, ws.ws_column);
      }

    // And copy only the real part of fft_src in dst
    for(int i = 0 ; i < ws.h_src; ++i)
      {
	for(int j = 0 ; j < ws.w_src ; ++j)
	  {
	    dst->data[i*ws.w_src + j] = ws.fft_copy->data[i*2*ws.w_res+2*j];
	  }
      }
  }

  /*********************************/
  /* Circular convolution with GSL   */
  /*********************************/

  void circular_convolution(gsl_matrix * src, gsl_matrix * kernel, gsl_matrix * dst)
  {
    assert(kernel->size1 <= src->size1 && kernel->size2 <= src->size2 && dst->size1 == src->size1 && dst->size2 == dst->size2);

    int height;
    int width;

    height = src->size1;
    width = src->size2;

    // Create some required matrices
    gsl_fft_complex_workspace * ws_column = gsl_fft_complex_workspace_alloc(height);
    gsl_fft_complex_workspace * ws_line = gsl_fft_complex_workspace_alloc(width);
    gsl_fft_complex_wavetable * wv_column = gsl_fft_complex_wavetable_alloc(height);
    gsl_fft_complex_wavetable * wv_line = gsl_fft_complex_wavetable_alloc(width);

    // We need to create a matrix holding 2 times the number of coefficients of src for the real and imaginary parts
    gsl_matrix * fft = gsl_matrix_alloc(height, 2*width);
    gsl_matrix * fft_copy = gsl_matrix_alloc(height, 2*width);

    gsl_matrix_set_zero(fft);
    gsl_vector_view xv, yv;
    // memcpy takes care of the strides,
    for(unsigned int i = 0 ; i < src->size2 ; ++i)
      {
	xv = gsl_matrix_subcolumn(fft,2*i,0, src->size1);
	yv = gsl_matrix_column(src, i);
	gsl_vector_memcpy(&xv.vector, &yv.vector);
      }

    // when zero-padding, we must ensure that the center of the kernel
    // is copied on the corners of the padded image
    // There are four quadrants to copy

    // The columns on the left of the center are put on the extreme right of fft_kernel
    for(unsigned int i = 0 ; i < int(kernel->size2/2) ; ++i)
      {
	// The top rows of kernel are put on the bottom rows of fft_kernel
	xv = gsl_matrix_subcolumn(fft,2*width-2*int(kernel->size2/2)+2*i+1,height - int(kernel->size1/2),int(kernel->size1/2));
	yv = gsl_matrix_subcolumn(kernel, i, 0, int(kernel->size1/2));
	gsl_vector_memcpy(&xv.vector,&yv.vector);
	// The bottom rows of the kernel are put on the top rows of fft_kernel
	xv = gsl_matrix_subcolumn(fft,2*width-2*int(kernel->size2/2)+2*i+1,0,int((kernel->size1+1)/2));
	yv = gsl_matrix_subcolumn(kernel, i, int(kernel->size1/2), int((kernel->size1+1)/2) );
	gsl_vector_memcpy(&xv.vector,&yv.vector);
      }
    // The columns on the right of the center are put on the extreme left of fft_kernel
    for(unsigned int i = int(kernel->size2/2) ; i < kernel->size2 ; ++i)
      {
	xv = gsl_matrix_subcolumn(fft,2*(i-int(kernel->size2/2))+1,height - int(kernel->size1/2),int(kernel->size1/2));
	yv = gsl_matrix_subcolumn(kernel, i, 0, int(kernel->size1/2));
	// The top rows of kernel are put on the bottom rows of fft_kernel
	gsl_vector_memcpy(&xv.vector,&yv.vector);
	// The bottom rows of the kernel are put on the top rows of fft_kernel
	xv = gsl_matrix_subcolumn(fft,2*(i-int(kernel->size2/2))+1,0,int((kernel->size1+1)/2));
	yv = gsl_matrix_subcolumn(kernel, i, int(kernel->size1/2), int((kernel->size1+1)/2));
	gsl_vector_memcpy(&xv.vector,&yv.vector);
      }

    // We compute the 2 forward DFT at once
    for(int i = 0 ; i < height ; ++i)
      {
	// Apply the FFT on the line i
	gsl_fft_complex_forward (&fft->data[i*2*width],1, width, wv_line, ws_line);
      }
    for(int j = 0 ; j < width ; ++j)
      {
	// Apply the FFT on the column j
	gsl_fft_complex_forward (&fft->data[2*j],width, height, wv_column, ws_column);
      }

    // Their element-wise product : be carefull, the matrices hold complex numbers !
    // We need a copy of fft to perform the product properly
    double re_h, im_h, re_hs, im_hs;
    for(int i = 0 ; i < height ; ++i)
      {
	for(int j = 0 ; j < width ; ++j)
	  {
	    re_h = gsl_matrix_get(fft,i, 2*j);
	    im_h = gsl_matrix_get(fft,i, 2*j+1);
	    re_hs = gsl_matrix_get(fft,(height-i)%height, ((2*width-2*j)%(2*width)));
	    im_hs = - gsl_matrix_get(fft,(height-i)%height, ((2*width-2*j)%(2*width))+1);

	    gsl_matrix_set(fft_copy, i, 2*j, 0.5*(re_h*im_h - re_hs*im_hs));
	    gsl_matrix_set(fft_copy, i, 2*j+1, -0.25*(re_h*re_h - im_h * im_h - re_hs * re_hs + im_hs * im_hs));
	  }
      }

    // And the inverse FFT, which is done in the similar way as before
    for(int i = 0 ; i < height ; ++i)
      {
	// Apply the FFT^{-1} on the line i
	gsl_fft_complex_inverse(&fft_copy->data[i*2*width],1, width, wv_line, ws_line);
      }

    for(int j = 0 ; j < width ; ++j)
      {
	// Apply the FFT^{-1} on the column j
	gsl_fft_complex_inverse(&fft_copy->data[2*j],width, height, wv_column, ws_column);
      }

    // And copy only the real part of fft_src in dst
    for(unsigned int i = 0 ; i < src->size2 ; ++i)
      {
	xv = gsl_matrix_subcolumn(fft_copy, 2*i,0,src->size1);
	gsl_matrix_set_col(dst, i, &xv.vector);
      }

    gsl_fft_complex_workspace_free(ws_column);
    gsl_fft_complex_workspace_free(ws_line);
    gsl_fft_complex_wavetable_free(wv_column);
    gsl_fft_complex_wavetable_free(wv_line);

    gsl_matrix_free(fft);
    gsl_matrix_free(fft_copy);
  }

  /******************************************************/
  /* Circular convolution with GSL with an optimal size   */
  /******************************************************/

  void circular_convolution_optimal(gsl_matrix * src, gsl_matrix * kernel, gsl_matrix * dst)
  {
    // In order to optimze the size for a circular convolution
    // it is sufficient to remind that a circular convolution is the restriction to the central part
    // of a linear convolution performed on a larger image with wrapped around values on the borders

    assert(kernel->size1 <= src->size1 && kernel->size2 <= src->size2 && dst->size1 == src->size1 && dst->size2 == dst->size2);

    int h_src = src->size1;
    int w_src = src->size2;
    int h_kernel = kernel->size1;
    int w_kernel = kernel->size2;

    // The size of the wrapped around image with enough pixel values on both sides (i.e. the size of the kernel)
    int h_wrapped = src->size1 + kernel->size1;
    int w_wrapped = src->size2 + kernel->size2;
    int height = find_closest_factor(h_wrapped, GSL_FACTORS);
    int width = find_closest_factor(w_wrapped, GSL_FACTORS);

    // Create some required matrices
    gsl_fft_complex_workspace * ws_column = gsl_fft_complex_workspace_alloc(height);
    gsl_fft_complex_workspace * ws_line = gsl_fft_complex_workspace_alloc(width);
    gsl_fft_complex_wavetable * wv_column = gsl_fft_complex_wavetable_alloc(height);
    gsl_fft_complex_wavetable * wv_line = gsl_fft_complex_wavetable_alloc(width);

    // We need to create a matrix holding 2 times the number of coefficients of src for the real and imaginary parts
    gsl_matrix * fft = gsl_matrix_alloc(height, 2*width);
    gsl_matrix * fft_copy = gsl_matrix_alloc(height, 2*width);

    // Copy and wrap around src
    // fft is filled differently for 10 regions (where the regions filled in with 0 is counted as a single region)
    // wrap bottom right | wrap bottom | wrap bottom left | 0
    //     wrap right    |     src     |   wrap left      | 0
    // wrap top  right   |  wrap top   |  wrap top left   | 0
    //        0          |      0      |       0          | 0

    gsl_matrix_set_zero(fft);
    int i_src, j_src;
    for(int i = 0 ; i < h_src + h_kernel ; ++i)
      {
	i_src = i - int((h_kernel+1)/2);
	if(i_src < 0)
	  i_src += h_src;
	else if(i_src >= h_src)
	  i_src -= h_src;
	for(int j = 0 ; j < w_src + w_kernel ; ++j)
	  {
	    j_src = j - int((w_kernel+1)/2);
	    if(j_src < 0)
	      j_src += w_src;
	    else if(j_src >= w_src)
	      j_src -= w_src;

	    fft->data[i * 2*width + 2*j] = src->data[i_src * src->size2 + j_src];
	  }
      }

    //////
    // Given this new source image, the following is exactly the same as for performing a linear convolution in GSL
    //////

    // when zero-padding, we must ensure that the center of the kernel
    // is copied on the corners of the padded image
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
	    fft->data[i_src * 2*width + 2*j_src+1] = kernel->data[i * w_kernel + j];
	  }
      }

    // We compute the 2 forward DFT at once
    for(int i = 0 ; i < height ; ++i)
      {
	// Apply the FFT on the line i
	gsl_fft_complex_forward (&fft->data[i*2*width],1, width, wv_line, ws_line);
      }
    for(int j = 0 ; j < width ; ++j)
      {
	// Apply the FFT on the column j
	gsl_fft_complex_forward (&fft->data[2*j],width, height, wv_column, ws_column);
      }

    // Their element-wise product : be carefull, the matrices hold complex numbers !
    // We need a copy of fft to perform the product properly
    double re_h, im_h, re_hs, im_hs;
    for(int i = 0 ; i < height ; ++i)
      {
	for(int j = 0 ; j < width ; ++j)
	  {
	    re_h = fft->data[i*2*width + 2*j];
	    im_h = fft->data[i*2*width + 2*j+1];
	    re_hs = fft->data[((height-i)%height)*2*width + ((2*width-2*j)%(2*width))];
	    im_hs = -fft->data[((height-i)%height)*2*width + ((2*width-2*j)%(2*width))+1];

	    fft_copy->data[i*2*width+2*j] = 0.5*(re_h*im_h - re_hs*im_hs);
	    fft_copy->data[i*2*width+2*j+1] = -0.25*(re_h*re_h - im_h * im_h - re_hs * re_hs + im_hs * im_hs);
	  }
      }

    // And the inverse FFT, which is done in the similar way as before
    for(int i = 0 ; i < height ; ++i)
      {
	// Apply the FFT^{-1} on the line i
	gsl_fft_complex_inverse(&fft_copy->data[i*2*width],1, width, wv_line, ws_line);
      }

    for(int j = 0 ; j < width ; ++j)
      {
	// Apply the FFT^{-1} on the column j
	gsl_fft_complex_inverse(&fft_copy->data[2*j],width, height, wv_column, ws_column);
      }

    // And copy only the real part of the central part of fft_src in dst
    for(int i = 0 ; i < h_src; ++i)
      for(int j = 0 ; j < w_src ; ++j)
	dst->data[i*w_src + j] = fft_copy->data[(i+int((h_kernel+1)/2))*2*width+2*(int((w_kernel+1)/2)+j)];

    gsl_fft_complex_workspace_free(ws_column);
    gsl_fft_complex_workspace_free(ws_line);
    gsl_fft_complex_wavetable_free(wv_column);
    gsl_fft_complex_wavetable_free(wv_line);

    gsl_matrix_free(fft);
    gsl_matrix_free(fft_copy);
  }

  void circular_convolution_fft_gsl_combined(gsl_matrix * src, gsl_matrix * kernel, gsl_matrix * dst)
  {
    if(is_optimal(src->size1, GSL_FACTORS) && is_optimal(src->size2,GSL_FACTORS))
      circular_convolution(src,kernel,dst);
    else
      circular_convolution_optimal(src,kernel,dst);
  }


  ////////////////// ICI !!!

void convolve(Workspace &ws, double * src, double * kernel, double * dst)
  {
    switch(ws.mode)
      {
      case LINEAR:
	gsl_matrix_view src_view = gsl_matrix_view_array(src, ws.h_src, ws.w_src);
	gsl_matrix_view kernel_view = gsl_matrix_view_array(kernel, ws.h_kernel, ws.w_kernel);
	gsl_matrix_view dst_view = gsl_matrix_view_array(dst, ws.h_src, ws.w_src);
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
