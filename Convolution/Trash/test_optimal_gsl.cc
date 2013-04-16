// g++ -o test_optimal_gsl test_optimal_gsl.cc 

#include <fstream>
#include <cstdio>
#include <cstdlib>


int GSL_FACTORS[7] = {7,6,4,5,3,2,0}; // end with zero to detect the end of the array

// Code taken from gsl/fft/factorize.c
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
    }

    if (n == 1)
    {
        factors[0] = 1;
        *n_factors = 1;
        return ;
    }

    /* deal with the implemented factors first */

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
    int nf;
    int factors[64];
    int i = 0;
    factorize(n, &nf, factors,implemented_factors);
    // We just have to check if the last factor belongs to GSL_FACTORS
    while(implemented_factors[i])
    {
        if(factors[nf-1] == implemented_factors[i])
	  {
	    if(n % (4*4*4*2) != 0)
	      return true;
	    else
	      return false;
	  }
	
        i++;
    }
    //printf("%i , %i \n", n, nf);
    
    return false;
}

int is_prime(int n, int * implemented_factors)
{
    int nf;
    int factors[64];
    int i = 0;
    factorize(n, &nf, factors,implemented_factors);
    // We just have to check if the last factor belongs to GSL_FACTORS
    return nf;  

}


int main(int argc, char * argv[])
{
  
  int bound = atoi(argv[1]);
  
  std::ofstream outfile("optimal_gsl.txt");
  

  for(int i = 2 ; i < bound ; ++i)
    {
      outfile << i << " " << is_optimal(i,GSL_FACTORS) << "\n";
    }
  outfile.close();
  

  int w,h;
  double t_unpad, t_pad, t_std;
  
  std::ifstream infile_gsl("benchmarks_CircularConvolution_gsl.txt");
  outfile.open("benchmarks_CircularConvolution_combined_gsl.txt");
  
  while(!infile_gsl.eof())
    {
      infile_gsl >> h;
      infile_gsl >> w;
      infile_gsl >> t_unpad;
      infile_gsl >> t_pad;
      infile_gsl >> t_std;
      if(is_optimal(h,GSL_FACTORS))
	{
	  outfile << h << " " << w << " " << t_unpad <<std::endl;
	}
      else
	outfile << h << " " << w << " " << t_pad <<std::endl;
      
    }
  outfile.close();
      



}
