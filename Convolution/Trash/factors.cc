// Compile with g++ -o factors factors.cc `pkg-config --libs --cflags fftw3 gsl` -lX11 -lpthread -O3 -Wall

#include <gsl/gsl_fft_complex.h>
#include <cstdio>
#include <cstdlib>

int GSL_FACTORS[7] = {7,6 ,5,4,3,2,0}; // end with zero to detect the end of the array
int FFTW_FACTORS[7] = {13,11,7,5,3,2,0};

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

  /* deal with any other even prime factors (there is only one) */

  if(ntest != 1)
  {
      factors[nf] = ntest;
      nf++;
  }
  // Why not, from this point, taking the reminder ????
  /*factor = 2;

  while ((ntest % factor) == 0 && (ntest != 1))
    {
      printf("Case factor 2\n");
      ntest = ntest / factor;
      factors[nf] = factor;
      nf++;
    }*/

  /* deal with any other odd prime factors */

  /*factor = 3;

  while (ntest != 1)
    {
      while ((ntest % factor) != 0)
        {
          factor += 2;
        }
      ntest = ntest / factor;
      factors[nf] = factor;
      nf++;
    }*/

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
    bool is_optimal = true;
    int i = 0;
    factorize(n, &nf, factors,implemented_factors);
    // We just have to check if the last factor belongs to GSL_FACTORS
    while(implemented_factors[i])
    {
        if(factors[nf-1] == implemented_factors[i])
            return true;
        i++;
    }
    return false;
}

int find_closest_factor(int n, int * implemented_factor)
{
    int j;
    if(is_optimal(n,implemented_factor))
    {
        return n;
    }
    else
    {
        j = n+1;
        while(!is_optimal(j,implemented_factor))
            ++j;
        return j;
    }
}

void display_factors(int n)
{
    int nf;
    int factors[64];
    factorize(n, &nf, factors,GSL_FACTORS);  
    printf("GSL : \n");
    
    for(int i = 0 ;  i< nf ; ++i)
      {
	printf("%i ",factors[i]);	
      }
    printf("\n");
    
   factorize(n, &nf, factors,FFTW_FACTORS);  
    printf("FFTW : \n");
    
    for(int i = 0 ;  i< nf ; ++i)
      {
	printf("%i ",factors[i]);	
      }
    printf("\n");    
}


int main(int argc, char * argv[])
{
  if(argc == 2)
    display_factors(atoi(argv[1]));
  else
    {
      
    int j;
    for(int i =1 ; i < 513 ; ++i)
    {
        printf("%i -> gsl : %i ; fftw : %i \n", i, find_closest_factor(i,GSL_FACTORS), find_closest_factor(i,FFTW_FACTORS));//find_closest_factor_fftw(i));
    }
    }
  
}
