// Compilation with :
// g++ -o convolution_octave convolution_octave.cc -DCONVOLUTION=0 -O3 -Wall `pkg-config --libs --cflags gsl fftw3`
// Convolution:
// - 0 : fftw, 
// - 1 : gsl, 
// - 2 : std

// Octave and C++ :
// see : http://www.mathias-michel.de/index.php?page=4&site=4


#if CONVOLUTION==0
#include "convolution_fftw.h"
using namespace FFTW_Convolution;

#elif CONVOLUTION==1
#include "convolution_gsl.h"
using namespace GSL_Convolution;

#elif CONVOLUTION==2
#include "convolution_std.h"
using namespace STD_Convolution;

#else
# error Unrecognized convolution type!
#endif

// What we do in this script is to randomly generate 1D and 2D sources and kernels
// to compute their convolution (full, same, valid, same_unpadded, circular_same, circular_same_padded)
// and compare the output with the outputs of Octave convn and fftconv functions

#define NB_CONVOLUTIONS_PER_CATEGORY 10

int main(int argc, char * argv[])
{

  Workspace ws;
  init_workspace(ws, CONVOLUTION_MODE, h_src, w_src, h_kernel, w_kernel);

  clear_workspace(ws);
}
