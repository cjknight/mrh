/* -*- c++ -*- */

#if !defined(_USE_MPI)

#ifndef MPI_H
#define MPI_H

#define MPI_COMM_WORLD 0

typedef int MPI_Comm;

void MPI_Init(int * argc, char *** argv);

void MPI_Comm_rank(int comm, int * rnk);
void MPI_Comm_size(int comm, int * nrnk);

void MPI_Barrier(int comm);

void MPI_Finalize();

#endif
#endif
