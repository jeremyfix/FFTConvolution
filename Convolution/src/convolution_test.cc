// Compilation with :
// FFTW : g++ -o convolution_test convolution_test.cc -DCONVOLUTION=0 -O3 -Wall `pkg-config --libs --cflags gsl fftw3` -std=c++11 `mkoctfile -p ALL_CXXFLAGS` `mkoctfile -p OCTAVE_LIBS` `mkoctfile -p LFLAGS`
// GSL : g++ -o convolution_test convolution_test.cc -DCONVOLUTION=1 -O3 -Wall `pkg-config --libs --cflags gsl fftw3` -std=c++11 `mkoctfile -p ALL_CXXFLAGS` `mkoctfile -p OCTAVE_LIBS` `mkoctfile -p LFLAGS`
// STD : g++ -o convolution_test convolution_test.cc -DCONVOLUTION=2 -O3 -Wall `pkg-config --libs --cflags gsl fftw3` -std=c++11 `mkoctfile -p ALL_CXXFLAGS` `mkoctfile -p OCTAVE_LIBS` `mkoctfile -p LFLAGS`

// Convolution:
// - 0 : fftw, 
// - 1 : gsl, 
// - 2 : std

// This requires that you have installed
// octave-headers
// octave-signal
// octave-image

// Results are displayed in the terminal.
// In case of errors, these are dumped in a convolution_octave.log


#if CONVOLUTION==0
#include "convolution_fftw.h"
using namespace FFTW_Convolution;
#define DO_PADDED_UNPADDED_TESTS true
#define SUFFIX_M_SCRIPT "fftw"

#elif CONVOLUTION==1
#include "convolution_gsl.h"
using namespace GSL_Convolution;
#define DO_PADDED_UNPADDED_TESTS true
#define SUFFIX_M_SCRIPT "gsl"

#elif CONVOLUTION==2
#include "convolution_std.h"
using namespace STD_Convolution;
#define DO_PADDED_UNPADDED_TESTS false
#define SUFFIX_M_SCRIPT "std"

#else
# error Unrecognized convolution type!
#endif

// What we do in this script is to randomly generate 1D and 2D sources and kernels
// to compute their convolution (full, same, valid, same_unpadded, circular_same, circular_same_padded)
// and compare the output with the outputs of Octave convn and fftconv functions
// We report if the results have the same dimensions and do not differ in Euclidean norm
// more than NORM_LIMIT_BEFORE_ERROR (1e-10)

#define NB_CONVOLUTIONS_PER_CATEGORY 50
#define NORM_LIMIT_BEFORE_ERROR 1e-10

#include <tuple>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>

// Octave part:
#include <octave/config.h>
#include <octave/dMatrix.h>
#include <octave/oct-convn.h>

// All these includes are for using fftconv
// indeed, the C++ API does not provide access to
// fftconv nor fftfilt
// So we rather write a m script and go through feval to evaluate it
#include <octave/oct.h>
#include <octave/octave.h>
#include <octave/parse.h>
#include <octave/toplev.h>


typedef std::tuple<int, int, int, int> size_tuple;

// Some utilitary functions to display colored text within the terminal
void print_error(std::string message)
{
  std::cout << std::setfill(' ') << std::setw(10) << ' ';
  printf("%c[%d;%dm %s %c[%dm\n",27,1,31,message.c_str(), 27,0);
}
void print_ok(std::string message)
{
  std::cout << std::setfill(' ') << std::setw(10) << ' ';
  printf("%c[%d;%dm %s %c[%dm\n",27,1,32,message.c_str(), 27,0);
}

void print_results(std::string convolution_name, int nb_success, int nb_trials)
{   
  std::ostringstream ostr;
  if(nb_success == nb_trials)
    {
      ostr.str("");
      ostr << "All tests passed for " << convolution_name << " " << nb_success << " / " << nb_trials;
      print_ok(ostr.str());
    }
  else
    {
      ostr.str("");
      ostr << "Some tests failed for "  << convolution_name << " " << nb_trials-nb_success << " / " << nb_trials << " failed";
      print_error(ostr.str());
    }
}

// This is an auxiliary function to run the convolutions and compare the results
bool bench_and_compare(Workspace &ws, int h_src, int w_src, int h_kernel, int w_kernel, double *src, double* kernel, Convolution_Mode mode, Matrix &src_mat, Matrix &kernel_mat, std::ofstream &logfile)
{
  // Perform the convolution with our implementations
  init_workspace(ws, mode, h_src, w_src, h_kernel, w_kernel);
  convolve(ws, src, kernel);

  Matrix tmp_matrix, dst_octave;
  octave_value_list functionArguments;
  octave_value_list result;
  switch(mode)
    {
    case LINEAR_FULL:
      logfile << "dst = convn(src, kernel, 'full');";
      dst_octave = convn(src_mat, kernel_mat, convn_full);
      break;
    case LINEAR_SAME:
      logfile << "dst = convn(src, kernel, 'same');";
      dst_octave = convn(src_mat, kernel_mat, convn_same);
      break;
    case LINEAR_VALID:
      logfile << "dst = convn(src, kernel, 'valid');";
      dst_octave = convn(src_mat, kernel_mat, convn_valid);
      if(ws.h_dst + ws.w_dst == 0)
	logfile << "Warning, the valid convolution results in an empty matrix " << std::endl;
      break;
#if DO_PADDED_UNPADDED_TESTS
    case LINEAR_SAME_UNPADDED:
      logfile << "dst = convn(src, kernel, 'same');";
      dst_octave = convn(src_mat, kernel_mat, convn_same);
      break;
#endif      
    case CIRCULAR_FULL:
#if DO_PADDED_UNPADDED_TESTS
    case CIRCULAR_FULL_UNPADDED:
#endif
      // perform the convolution with Octave
      functionArguments (0) = src_mat;
      functionArguments (1) = kernel_mat;
      if(h_src == 1 || w_src == 1)
	{
	  result = feval ("fftconv", functionArguments, 1);
	  logfile << "dst = fftconv(src, kernel);" << std::endl;
	}
      else
	{
	  result = feval ("fftconv2", functionArguments, 1);
	  logfile << "dst = fftconv2(src, kernel);" << std::endl;
	}

      dst_octave = result(0).matrix_value ();
      break;
    case CIRCULAR_SAME:
#if DO_PADDED_UNPADDED_TESTS
    case CIRCULAR_SAME_PADDED: 
#endif   
      // perform the convolution with Octave
      functionArguments (0) = src_mat;
      functionArguments (1) = kernel_mat;
      if(h_src == 1 || w_src == 1)
	{
	  result = feval ("fftconv", functionArguments, 1);
	  logfile << "dst = fftconv(src, kernel);" << std::endl;
	}
      else
	{
	  result = feval ("fftconv2", functionArguments, 1);
	  logfile << "dst = fftconv2(src, kernel);" << std::endl;
	}
      // We extract a subpart of the matrix
      tmp_matrix = result(0).matrix_value ();

      dst_octave = Matrix(h_src, w_src);
      for(int i = 0 ; i < h_src ; ++i)
	for(int j = 0 ; j < w_src ; ++j)
	  dst_octave(i,j) = tmp_matrix(i,j);

      break;
      /*
#if DO_PADDED_UNPADDED_TESTS
    case CIRCULAR_FULL_PADDED:
      // perform the convolution with Octave
      functionArguments (0) = src_mat;
      functionArguments (1) = kernel_mat;
      if(h_src == 1 || w_src == 1)
	{
	  result = feval ("fftconv", functionArguments, 1);
	  logfile << "dst = fftconv(src, kernel);" << std::endl;
	}
      else
	{
	  result = feval ("fftconv2", functionArguments, 1);
	  logfile << "dst = fftconv2(src, kernel);" << std::endl;
	}

      dst_octave = result(0).matrix_value ();
      break;
    case CIRCULAR_SAME_PADDED:    
      // perform the convolution with Octave
      functionArguments (0) = src_mat;
      functionArguments (1) = kernel_mat;
      if(h_src == 1 || w_src == 1)
	{
	  result = feval ("fftconv", functionArguments, 1);
	  logfile << "dst = fftconv(src, kernel);" << std::endl;
	}
      else
	{
	  result = feval ("fftconv2", functionArguments, 1);
	  logfile << "dst = fftconv2(src, kernel);" << std::endl;
	}
      // We extract a subpart of the matrix
      tmp_matrix = result(0).matrix_value ();

      dst_octave = Matrix(h_src, w_src);
      for(int i = 0 ; i < h_src ; ++i)
	for(int j = 0 ; j < w_src ; ++j)
	  dst_octave(i,j) = tmp_matrix(i,j);
      break;
#endif 
      */
    default:
      print_error("Unrecognized convolution mode");
      std::cout << "Mode : " << mode << std::endl;
      return false;
    }


  // We now compare the results
  if( ((ws.h_dst + ws.w_dst == 0) && (int(dst_octave.dim1()) != 0 && int(dst_octave.dim2()) != 0)) ||// dst_octave.dim1 == 0 means empty matrix
      ((ws.h_dst != dst_octave.dim1() || ws.w_dst != dst_octave.dim2())&& (int(dst_octave.dim1()) != 0 && int(dst_octave.dim2()) != 0)) )
    {
      std::cout << "Dimensions " << ws.h_dst + ws.w_dst << " != " << int(dst_octave.dim1()) << std::endl;
      // Dump the error in the logfile
      logfile << "Different sizes !! (" << ws.h_dst << "," << ws.w_dst << ") != (" << dst_octave.dim1() << "," << dst_octave.dim2()  << ")" << std::endl;
      logfile << "FAILED" << std::endl;
      clear_workspace(ws);
      return false;
    }
  else
    {
      // Check the euclidean norm of the results
      double dist = 0.0;
      for(int r = 0 ; r < ws.h_dst ; ++r)
	for(int c = 0 ; c < ws.w_dst ; ++c)
	  dist += (ws.dst[r * ws.w_dst + c] - dst_octave(r, c))*(ws.dst[r * ws.w_dst + c] - dst_octave(r, c));
      dist = sqrt(dist);
      if(dist > NORM_LIMIT_BEFORE_ERROR)
	{  
	  // Dump the error in the logfile
	  logfile << "The results differ of more than " <<  NORM_LIMIT_BEFORE_ERROR << " in euclidean norm , norm=" << dist ;
	  // We dump the two results
	  logfile << "Output from our libraries :" << std::endl;
	  logfile << "dst = [";
	  for(int r = 0 ; r < ws.h_dst; ++r)
	    {
	      for(int c = 0 ; c < ws.w_dst ; ++c)
		logfile << ws.dst[r*ws.w_dst + c] <<  " ";
	      if(r != ws.h_dst - 1)
		logfile << ";";
	      else
		logfile << "];" << std::endl;
	    }
	  logfile << "Output from octave : " << std::endl;
	  logfile << "dst_octave = [";
	  for(int r = 0 ; r < ws.h_dst; ++r)
	    {
	      for(int c = 0 ; c < ws.w_dst ; ++c)
		logfile << dst_octave(r, c) <<  " ";
	      if(r != ws.h_dst - 1)
		logfile << ";";
	      else
		logfile << "];" << std::endl;
	    }
	  clear_workspace(ws);
	  logfile << "FAILED" << std::endl;
	  return false;
	}
      else
	{
	  clear_workspace(ws);
	  logfile << "PASSED" << std::endl;
	  return true;
	}
    }

  clear_workspace(ws);
  printf("SHOULD NEVER OCCUR !!!");
  return true;
}

// For the circular convolution same and same_padded, Octave does not seem to handle
// circular convolutions modulo N, so we better write a matlab m file to make the comparison
void write_to_matlab_file(Workspace &ws, int h_src, int w_src, int h_kernel, int w_kernel, double *src, double* kernel, Convolution_Mode mode, std::ofstream &mfile, std::string category_name)
{
  // Perform the convolution with our implementations
  init_workspace(ws, mode, h_src, w_src, h_kernel, w_kernel);
  convolve(ws, src, kernel);

  // We now fill in the m-file with the code for performing the convolution
  mfile << "if(~ map_results.isKey('" << category_name << "_invalid_sizes'))" << std::endl;
  mfile << "map_results('" << category_name << "_invalid_sizes') = 0;" << std::endl;
  mfile << "map_results('" << category_name << "_failed') = 0;" << std::endl;
  mfile << "map_results('" << category_name << "_success') = 0;" << std::endl;
  mfile << "end" << std::endl;

  // we write in the matrix we convolve
  mfile << std::scientific ;
  mfile << "src = [";
  for(int i = 0 ; i < h_src ; ++i)
    {
      for(int j = 0 ; j < w_src ; ++j)
	mfile << src[i*w_src + j] << " ";
      if(i < h_src - 1)
	mfile << ";";
      else
	mfile << "];" << std::endl;
    }
  mfile << "kernel = [";
  for(int i = 0 ; i < h_kernel ; ++i)
    {
      for(int j = 0 ; j < w_kernel ; ++j)
	mfile << kernel[i*w_kernel + j] << " ";
      if(i < h_kernel - 1)
	mfile << ";" ;
      else
	mfile << "];" << std::endl;
    }
  mfile << "dst = [";
  for(int i = 0 ; i < ws.h_dst ; ++i)
    {
      for(int j = 0 ; j < ws.w_dst ; ++j)
	mfile << ws.dst[i*ws.w_dst + j] << " ";
      if(i < ws.h_dst - 1)
	mfile << ";" ;
      else
	mfile << "];" << std::endl;
    }
  if(h_src == 1 || w_src == 1)
    {
      mfile << "dst_matlab = cconv(src, kernel, " << h_src*w_src << ");"<< std::endl;
    }
  else
    {
      // to see ...
    }
  // Check the norm of the difference
  mfile << "if(size(dst) ~= size(dst_matlab))" << std::endl;
  mfile << "map_results('" << category_name << "_invalid_sizes') = " << "map_results('" << category_name << "_invalid_sizes') + 1;" << std::endl;
  mfile << "map_results('" << category_name << "_failed') = " << "map_results('" << category_name << "_failed') + 1;" << std::endl;
  mfile << " msg = sprintf('Invalid sizes : %i x %i ~= %i x %i\\n', size(dst, 1), size(dst, 2), size(dst_matlab,1), size(dst_matlab,2));" << std::endl;
  mfile << " disp(msg);" << std::endl;
  mfile << "else " << std::endl;

  // Apparently, we cannot get better than around 2*1e-5 difference in the norm
  mfile << "if(norm(dst - dst_matlab) > 3e-5)" << std::endl;
  mfile << " msg = sprintf('Norm of the difference : %e > 3e-5\\n', norm(dst - dst_matlab));" << std::endl;
  mfile << " disp(msg);" << std::endl;
  mfile << "map_results('" << category_name << "_failed') = " << "map_results('" << category_name << "_failed') + 1;" << std::endl;
  mfile << "else " << std::endl;
  mfile << "map_results('" << category_name << "_success') = " << "map_results('" << category_name << "_success') + 1;" << std::endl;
  //mfile << "disp('Ok!')" << std::endl;
  mfile << "end" << std::endl;
  mfile << "end" << std::endl;
  
}


// In the main, we build the signals for which we want to test the convolutions
// and then call our auxiliary function to perform the test
int main(int argc, char * argv[])
{

  // In these benchmarks, we consider various combinations of kernel sizes
  // 1D: {even/odd} source size, {even/odd} kernel size -> 4 conditions
  //  -> We test with (h, 1) and (1, w) sizes           -> 4 conditions
  // 2D: {even/odd} source size, {even/odd} kernel size -> 4 conditions
  // Each of these conditions is tested NB_CONVOLUTIONS_PER_CATEGORY for each mode :
  // - linear full, 
  // - linear same, 
  // - linear valid
  // - linear same unpadded,
  // - circular full
  // In a matlab file, we provide what to test:
  // - circular same
  // - circular same padded

  // We initialize the octave evalutor
  octave_main (0, argv, 1);

  srand(time(NULL));

  int h_src, w_src, h_kernel, w_kernel;
  
  std::vector< std::string > category_labels;
  std::vector< std::vector< size_tuple > > category_sizes;

  double * src, * kernel;
  src = new double[3000]; // We allocate a vector sufficiently large to fit for all the conditions
  kernel = new double[3000];

  Workspace ws;

  std::string logfilename, mfilename;
  std::ostringstream ostr("");
  ostr << "convolution_test_" << SUFFIX_M_SCRIPT << ".log";
  logfilename = ostr.str();

  ostr.str("");
  ostr << "convolution_test_" << SUFFIX_M_SCRIPT << ".m";
  mfilename = ostr.str();


  std::ofstream logfile(logfilename);
  std::ofstream mfile(mfilename);

  // Create the map which will contain the summary of the tests
  mfile << "map_results = containers.Map();" << std::endl;

  category_labels.push_back("1D even/even Hx1");
  category_sizes.push_back(std::vector< size_tuple>());
  for(int i = 0 ; i < NB_CONVOLUTIONS_PER_CATEGORY; ++i)
    {
      h_src = 2*int(20.0 * rand()/double(RAND_MAX) + 30.0); // 2*[30 ; 50]
      w_src = 1;
      h_kernel = 2*int(20.0 * rand()/double(RAND_MAX) + 30.0); // 2*[30; 50]
      w_kernel = 1;
      category_sizes[category_sizes.size() - 1].push_back(std::make_tuple(h_src, w_src, h_kernel, w_kernel));
    }
  category_labels.push_back("1D even/odd Hx1");
  category_sizes.push_back(std::vector< size_tuple>());
  for(int i = 0 ; i < NB_CONVOLUTIONS_PER_CATEGORY; ++i)
    {
      h_src = 2*int(20.0 * rand()/double(RAND_MAX) + 30.0); // 2*[30 ; 50]
      w_src = 1;
      h_kernel = 2*int(20.0 * rand()/double(RAND_MAX) + 20.0)+1; // 2*[10; 30]+1
      w_kernel = 1;
      category_sizes[category_sizes.size() - 1].push_back(std::make_tuple(h_src, w_src, h_kernel, w_kernel));
    }

  category_labels.push_back("1D odd/even Hx1");
  category_sizes.push_back(std::vector< size_tuple>());
  for(int i = 0 ; i < NB_CONVOLUTIONS_PER_CATEGORY; ++i)
    {
      h_src = 2*int(20.0 * rand()/double(RAND_MAX) + 30.0)+1; // 2*[30 ; 50]
      w_src = 1;
      h_kernel = int(20.0 * rand()/double(RAND_MAX) + 20.0); // 2*[10; 30]+1
      w_kernel = 1;
      category_sizes[category_sizes.size() - 1].push_back(std::make_tuple(h_src, w_src, h_kernel, w_kernel));
    }

  category_labels.push_back("1D odd/odd Hx1");
  category_sizes.push_back(std::vector< size_tuple>());
  for(int i = 0 ; i < NB_CONVOLUTIONS_PER_CATEGORY; ++i)
    {
      h_src = 2*int(20.0 * rand()/double(RAND_MAX) + 30.0)+1; // 2*[30 ; 50]
      w_src = 1;
      h_kernel = int(20.0 * rand()/double(RAND_MAX) + 20.0)+1; // 2*[10; 30]+1
      w_kernel = 1;
      category_sizes[category_sizes.size() - 1].push_back(std::make_tuple(h_src, w_src, h_kernel, w_kernel));
    }


  // The 1D , 1xW  signals
  category_labels.push_back("1D even/even 1xW");
  category_sizes.push_back(std::vector< size_tuple>());
  for(int i = 0 ; i < NB_CONVOLUTIONS_PER_CATEGORY; ++i)
    {
      h_src = 1;
      w_src = 2*int(20.0 * rand()/double(RAND_MAX) + 30.0); // 2*[30 ; 50]
      h_kernel = 1;
      w_kernel = 2*int(20.0 * rand()/double(RAND_MAX) + 20.0); // 2*[10; 30]
      category_sizes[category_sizes.size() - 1].push_back(std::make_tuple(h_src, w_src, h_kernel, w_kernel));
    }
  category_labels.push_back("1D even/odd 1xW");
  category_sizes.push_back(std::vector< size_tuple>());
  for(int i = 0 ; i < NB_CONVOLUTIONS_PER_CATEGORY; ++i)
    {
      h_src = 1;
      w_src = 2*int(20.0 * rand()/double(RAND_MAX) + 30.0); // 2*[30 ; 50]
      h_kernel = 1;
      w_kernel = 2*int(20.0 * rand()/double(RAND_MAX) + 20.0)+1; // 2*[10; 30]+1
      category_sizes[category_sizes.size() - 1].push_back(std::make_tuple(h_src, w_src, h_kernel, w_kernel));
    }

  category_labels.push_back("1D odd/even 1xW");
  category_sizes.push_back(std::vector< size_tuple>());
  for(int i = 0 ; i < NB_CONVOLUTIONS_PER_CATEGORY; ++i)
    {
      h_src = 1;
      w_src = 2*int(20.0 * rand()/double(RAND_MAX) + 30.0)+1; // 2*[30 ; 50]
      h_kernel = 1;
      w_kernel = int(20.0 * rand()/double(RAND_MAX) + 20.0); // 2*[10; 30]+1
      category_sizes[category_sizes.size() - 1].push_back(std::make_tuple(h_src, w_src, h_kernel, w_kernel));
    }

  category_labels.push_back("1D odd/odd 1xW");
  category_sizes.push_back(std::vector< size_tuple>());
  for(int i = 0 ; i < NB_CONVOLUTIONS_PER_CATEGORY; ++i)
    {
      h_src = 1;
      w_src = 2*int(20.0 * rand()/double(RAND_MAX) + 30.0)+1; // 2*[30 ; 50]
      h_kernel = 1;
      w_kernel = int(20.0 * rand()/double(RAND_MAX) + 20.0)+1; // 2*[10; 30]+1;
      category_sizes[category_sizes.size() - 1].push_back(std::make_tuple(h_src, w_src, h_kernel, w_kernel));
    }
  

  // 2D signals
  category_labels.push_back("2D");
  category_sizes.push_back(std::vector< size_tuple>());
  for(int i = 0 ; i < NB_CONVOLUTIONS_PER_CATEGORY; ++i)
    {
      h_src = int(20.0 * rand()/double(RAND_MAX) + 30.0);
      w_src = int(20.0 * rand()/double(RAND_MAX) + 30.0); // [30 ; 50]
      h_kernel = int(20.0 * rand()/double(RAND_MAX) + 20.0); // [10; 30];
      w_kernel = int(20.0 * rand()/double(RAND_MAX) + 20.0); // [10; 30];
      category_sizes[category_sizes.size() - 1].push_back(std::make_tuple(h_src, w_src, h_kernel, w_kernel));
    }


  int nb_success_full, nb_success_valid, nb_success_same, nb_success_same_unpadded;
  int nb_success_circular_full, nb_success_circular_full_unpadded;

  for(unsigned int i = 0 ; i < category_sizes.size() ; ++i)
    {
      
      std::string label_category = category_labels[i];
      std::cout << "Test for category " << label_category << std::endl;
      nb_success_full = nb_success_valid = nb_success_same = nb_success_same_unpadded =0;
      nb_success_circular_full = nb_success_circular_full_unpadded = 0;

      for(unsigned int j = 0 ; j < category_sizes[i].size() ; ++j)
	{
	  std::tie (h_src, w_src, h_kernel, w_kernel) = category_sizes[i][j];	  
	  // We initialize the source
	  for(int i = 0 ; i < h_src * w_src ; ++i)
	    src[i] = rand()/double(RAND_MAX);
	  // And the kernel
	  for(int i = 0 ; i < h_kernel * w_kernel ; ++i)
	    kernel[i] = rand()/double(RAND_MAX);

	  // We now ask Octave to perform the convolution
	  // We build and fill in the matrices
	  Matrix src_mat(h_src, w_src), kernel_mat(h_kernel, w_kernel);
	  for(int r = 0 ; r < h_src ; ++r)
	    for(int c = 0 ; c < w_src ; ++c)
	      src_mat(r, c) = src[r * w_src + c];
	  for(int r = 0 ; r < h_kernel ; ++r)
	    for(int c = 0 ; c < w_kernel ; ++c)
	      kernel_mat(r, c) = kernel[r * w_kernel + c];

	  // We dump in the logfile the test we tried
	  logfile << "src = [ ";
	  for(int r = 0 ; r < h_src ; ++r)
	    {
	      for(int c = 0 ; c < w_src ; ++c)
		logfile << src_mat(r, c) << " ";
	      if(r == h_src - 1)
		logfile << "];" << std::endl;
	      else
		logfile << "; ";
	    }
	  logfile << "kernel = [ ";
	  for(int r = 0 ; r < h_kernel ; ++r)
	    {
	      for(int c = 0 ; c < w_kernel ; ++c)
		logfile << kernel_mat(r, c) << " ";
	      if(r == h_kernel - 1)
		logfile << "];" << std::endl;
	      else
		logfile << "; ";
	    }

	  // Do the tests
	  nb_success_full += bench_and_compare(ws, h_src, w_src, h_kernel, w_kernel, src, kernel, LINEAR_FULL, src_mat, kernel_mat, logfile);
	  nb_success_same += bench_and_compare(ws, h_src, w_src, h_kernel, w_kernel, src, kernel, LINEAR_SAME, src_mat, kernel_mat, logfile);
	  nb_success_valid += bench_and_compare(ws, h_src, w_src, h_kernel, w_kernel, src, kernel, LINEAR_VALID, src_mat, kernel_mat, logfile);
	  nb_success_circular_full += bench_and_compare(ws, h_src, w_src, h_kernel, w_kernel, src, kernel, CIRCULAR_FULL, src_mat, kernel_mat, logfile);

#if DO_PADDED_UNPADDED_TESTS
	  // Theses tests should be done only with fftw and gsl, not std
	  nb_success_same_unpadded += bench_and_compare(ws, h_src, w_src, h_kernel, w_kernel, src, kernel, LINEAR_SAME_UNPADDED, src_mat, kernel_mat, logfile);
	  nb_success_circular_full_unpadded += bench_and_compare(ws, h_src, w_src, h_kernel, w_kernel, src, kernel, CIRCULAR_FULL_UNPADDED, src_mat, kernel_mat, logfile);
#endif


	  // A matlab script is created for checking the results
	  // of the modulor N circular convolutions
	  // here we consider only the, so called , circular convolution same, i.e;
	  // a circular convolution modulor the size of the source image.
	  write_to_matlab_file(ws, h_src, w_src, h_kernel, w_kernel, src, kernel, CIRCULAR_SAME, mfile,label_category);
	  //write_to_matlab_file(ws, h_src, w_src, h_kernel, w_kernel, src, kernel, CIRCULAR_SAME_PADDED, mfile,label_category);


	}

      print_results("linear full", nb_success_full, NB_CONVOLUTIONS_PER_CATEGORY);
      print_results("linear_same", nb_success_same, NB_CONVOLUTIONS_PER_CATEGORY);
      print_results("linear valid", nb_success_valid, NB_CONVOLUTIONS_PER_CATEGORY);
      print_results("circular full", nb_success_circular_full, NB_CONVOLUTIONS_PER_CATEGORY);

#if DO_PADDED_UNPADDED_TESTS
      print_results("linear same unpadded" , nb_success_same_unpadded, NB_CONVOLUTIONS_PER_CATEGORY);
      print_results("circular full unpadded" , nb_success_circular_full_unpadded, NB_CONVOLUTIONS_PER_CATEGORY);
#endif
    }

  std::cout << std::setfill('-') << std::setw(50) << '-' << std::endl;

  ostr.str("");
  ostr << "To check the results of the circular convolutions (with a result of the same size as the source), please run the " << mfilename << " matlab file ";
  print_error(ostr.str());
  std::cout << std::setfill('-') << std::setw(50) << '-' << std::endl;


  // Finish by filling in the m-file the code to display the summary of the results
  mfile << "k = map_results.keys();" << std::endl;
  mfile << "v = map_results.values();" << std::endl;
  mfile << "for i=1:length(k)" << std::endl;
  mfile << " msg  = sprintf('%s : %i\\n', k{i}, v{i});" << std::endl;
  mfile << " disp(msg);" << std::endl;
  mfile << "end" << std::endl;

  delete[] src;
  delete[] kernel;
  logfile.close();
  mfile.close();
}
