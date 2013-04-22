#ifndef CONVOLUTION_STD_H
#define CONVOLUTION_STD_H

#include <cmath>

namespace STD_Convolution
{

  typedef enum
  {
    LINEAR,
    CIRCULAR
  } STD_Convolution_Mode;

  typedef struct Workspace
  {
    int h_src, w_src;
    int h_kernel, w_kernel;
    STD_Convolution_Mode mode;
  } Workspace;

  void init_workspace(Workspace & ws, STD_Convolution_Mode mode, int h_src, int w_src, int h_kernel, int w_kernel)
  {
    ws.h_src = h_src;
    ws.w_src = w_src;
    ws.h_kernel = h_kernel;
    ws.w_kernel = w_kernel;
    ws.mode = mode;
  }

  void clear_workspace(Workspace & ws)
  {
  }

  void update_workspace(Workspace & ws, STD_Convolution_Mode mode, int h_src, int w_src, int h_kernel, int w_kernel)
  {
    clear_workspace(ws);
    init_workspace(ws, mode, h_src, w_src, h_kernel, w_kernel);
  }

  /* ******************************************************************************************************** */
  /*     Standard  linear convolution, returns a result of the same size as the original src signal           */
  /* ******************************************************************************************************** */

  void linear_convolution(Workspace &ws, double * src, double * kernel, double * dst)
  {
    int i,j,k,l;
    double temp;
    int low_k, high_k, low_l, high_l;

    // For each pixel in the dest image
    for (i = 0 ; i < ws.h_src ; i++)
      {
	low_k = std::max(0, i - int((ws.h_kernel-1)/2));
	high_k = std::min(ws.h_src-1, i + int(ws.h_kernel/2));

	for (j = 0 ; j  < ws.w_src ; j++)
	  {
	    low_l = std::max(0, j - int((ws.w_kernel-1)/2));
	    high_l = std::min(ws.w_src-1, j + int(ws.w_kernel/2));
	    temp = 0.0;
	    // We browse the kernel
	    for (k = low_k ; k <= high_k ; k++)
	      {
		for(l = low_l ; l <= high_l ; l++)
		  {
		    temp += src[k*ws.w_src+l]* kernel[(i+int(ws.h_kernel/2)-k)*ws.w_kernel+(j+int(ws.w_kernel/2)-l)];
		  }
	      }
	    dst[i*ws.w_src+j] = temp;
	  }
      }
  }

  /* ******************************************************************************************************** */
  /*     Standard circular convolution, returns a result of the same size as the original src signal           */
  /* ******************************************************************************************************** */

  void circular_convolution(Workspace &ws, double * src, double * kernel, double * dst)
  {
    int i,j,k,l;
    double temp;

    int i_src, j_src;
    for (i = 0 ; i < ws.h_src ; ++i)
      {
	for (j = 0 ; j  < ws.w_src ; ++j)
	  {
	    temp = 0.0;
	    // We browse the kernel
	    for (k = 0 ; k < ws.h_kernel  ; ++k)
	      {
		i_src = i - k;
		if(i_src < 0)
		  i_src += ws.h_src;
		else if(i_src >= ws.h_src)
		  i_src -= ws.h_src;

		for(l = 0 ; l < ws.w_kernel ; ++l)
		  {
		    j_src = j - l;
		    if(j_src < 0)
		      j_src += ws.w_src;
		    else if(j_src >= ws.w_src)
		      j_src -= ws.w_src;

		    temp += src[i_src*ws.w_src + j_src]* kernel[k*ws.w_kernel+l];
		  }
	      }
	    dst[i*ws.w_src+j] = temp;
	  }
      }
  }

  void convolve(Workspace &ws, double * src, double * kernel, double * dst)
  {
    switch(ws.mode)
      {
      case LINEAR:
	linear_convolution(ws, src, kernel, dst);
	break;
      case CIRCULAR:
	circular_convolution(ws, src, kernel, dst);
	break;
      default:
	printf("Unrecognized convolution mode, possible modes are :\n");
	printf("   - LINEAR \n");
	printf("   - CIRCULAR \n");
	// TODO EXCEPTION
      }
  }


}

  

#endif
