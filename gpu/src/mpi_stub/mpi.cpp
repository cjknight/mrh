/* -*- c++ -*- */

#if !defined(_USE_MPI)

#include <cstdio>
#include <cstdlib>

#include "mpi.h"
  
/* ---------------------------------------------------------------------- */

void MPI_Init(int * argc, char *** argv) {};
  
/* ---------------------------------------------------------------------- */

void MPI_Comm_rank(int comm, int * rnk)
{
  if(comm != 0) {
    printf("LIBGPU :: MPI_Comm_rank() :: Error: comm != 0  are you trying to run MPI-enabled run with the mpi_stub.a library?\n");
    exit(1);
  }
  
  *rnk = 0;	
};

/* ---------------------------------------------------------------------- */

void MPI_Comm_size(int comm, int * nrnk)
{
  if(comm != 0) {
    printf("LIBGPU :: MPI_Comm_rank() :: Error: comm != 0  are you trying to run MPI-enabled run with the mpi_stub.a library?\n");
    exit(1);
  }
    
  *nrnk = 1;	
};

/* ---------------------------------------------------------------------- */

void MPI_Barrier(int comm) {};

/* ---------------------------------------------------------------------- */

void MPI_Finalize() {};

/* ---------------------------------------------------------------------- */

#endif
