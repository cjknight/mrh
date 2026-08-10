/* -*- c++ -*- */

#include <stdio.h>

#include "device.h"

#define _NUM_SIMPLE_TIMER 40
#define _NUM_SIMPLE_COUNTER 30
#include <unistd.h>
#include <string.h>
#include <sched.h>
#define _MIN(A,B) (A<B)?A:B
#define _MAX(A,B) (A>B)?A:B
#define _SIZE_FCI_BATCHES 6

/* ---------------------------------------------------------------------- */

void Device::mgpu_bcast(std::vector<double *> d_ptr, double * h_ptr, size_t size)
{
  // push data from host to first device
  
  pm->dev_set_device(0);
    
  int err = pm->dev_push_async(d_ptr[0], h_ptr, size);
  
  if(err) {
    printf("LIBGPU:: dev_push_async(d_ptr[0]) failed\n");
    exit(1);
  }

  for(int i=1; i<d_ptr.size(); ++i)
    pm->dev_memcpy_peer(d_ptr[i], i, d_ptr[0], 0, size);
}

/* ---------------------------------------------------------------------- */

void Device::mgpu_reduce(std::vector<double *> d_ptr, double * h_ptr, int N, bool blocking, std::vector<double *> buf_ptr, std::vector<int> active)
{
#if defined(_DEBUG_P2P)
  printf("LIBGPU :: -- GPU-GPU Reduction  Starting!\n");
#endif
  
  size_t size = N * sizeof(double);

  int num_active = 0;
  for(int i=0; i<num_devices; ++i) num_active += active[i];
  
  int nrecv = num_active / 2;

  int nactive = num_active;

  // accumulate result to device 0 using binary tree reduction
  
  int il = 0;
  while(nrecv > 0) {

#if defined(_DEBUG_P2P)
    printf("LIBGPU :: -- GPU-GPU Reduction  il= %i  nactive= %i  nrecv= %i\n",il,nactive,nrecv);
#endif
    
    // odd number of recievers and not last level (clean-up pre-reduction)
      
    if((nactive > 1) && (nactive % 2)) {
      
#if defined(_DEBUG_P2P)
      printf("LIBGPU :: -- GPU-GPU Reduction  pre clean-up odd reciever  nactive= %i  nrecv= %i\n",nactive,nrecv);
#endif
      
      int dest = nactive - 2;
      int src = nactive - 1;
      
      if(d_ptr[src] && active[src]) {
#if defined(_DEBUG_P2P)
	printf("LIBGPU :: -- GPU-GPU Reduction  -- src %i(%p) --> dest %i(%p, %p)\n",
	       src, d_ptr[src], dest, buf_ptr[dest], d_ptr[dest]);
#endif
	
	if(blocking) {
	  // need to ensure dest is done using buf
	  
	  pm->dev_set_device(dest);
	  
	  pm->dev_stream_wait();
	}
	
	// src initiates transfer
	
	pm->dev_set_device(src);
	
	pm->dev_memcpy_peer(buf_ptr[dest],dest, d_ptr[src], src, size);
	
	// dest launches kernel
	
	pm->dev_set_device(dest); 
	
	vecadd(buf_ptr[dest], d_ptr[dest], N);
      }
      
      nactive--;
    }
    
    // binary tree reduction
    
    if(nactive > nrecv) {

#if defined(_DEBUG_P2P)
      printf("LIBGPU :: -- GPU-GPU Reduction  binary reduction   nactive= %i  nrecv= %i\n",nactive,nrecv);
#endif
      
      int nsend = nactive - nrecv;

      for(int i=0; i<nsend; ++i) {

	int dest = i;
	int src = nrecv + i;

	if(d_ptr[src] && active[src]) {
#if defined(_DEBUG_P2P)	
	printf("LIBGPU :: -- GPU-GPU Reduction  -- src %i(%p) --> dest %i(%p, %p)\n",
	       src, d_ptr[src], dest, buf_ptr[dest], d_ptr[dest]);
#endif

	  if(blocking) {
	    // need to ensure dest is done using buf
	    
	    pm->dev_set_device(dest);
	    
	    pm->dev_stream_wait();
	  }

	  // src initiates transfer
	  
	  pm->dev_set_device(src);
	  
	  pm->dev_memcpy_peer(buf_ptr[dest], dest, d_ptr[src], src, size);

	  // dest launches kernel
	  
	  pm->dev_set_device(dest); 
	  
	  vecadd(buf_ptr[dest], d_ptr[dest], N);
	}
      }

      nactive = nrecv;

      // odd number of recievers and not last level (clean-up post-reduction)
      
      if((nrecv > 1) && (nrecv % 2)) {

#if defined(_DEBUG_P2P)
       	printf("LIBGPU :: -- GPU-GPU Reduction  post clean-up odd reciever  nactive= %i  nrecv= %i\n",nactive,nrecv);
#endif
    
	int dest = nrecv - 2;
	int src = nrecv - 1;

	if(d_ptr[src] && active[src]) {
#if defined(_DEBUG_P2P)	
	printf("LIBGPU :: -- GPU-GPU Reduction  -- src %i(%p) --> dest %i(%p, %p)\n",
	       src, d_ptr[src], dest, buf_ptr[dest], d_ptr[dest]);
#endif
	  if(blocking) {
	    // need to ensure dest is done using buf
	    pm->dev_set_device(dest);
	    
	    pm->dev_stream_wait();
	  }

	  // src initiates transfer
	  
	  pm->dev_set_device(src);
	  
	  pm->dev_memcpy_peer(buf_ptr[dest], dest, d_ptr[src], src, size);

	  // dest launches kernel
	  
	  pm->dev_set_device(dest); 
	  
	  vecadd(buf_ptr[dest], d_ptr[dest], N);
	}

	nrecv--;
	nactive--;
      }
      
    }

    nrecv /= 2;
    il++;
  }

  // accumulate result on host

#if defined(_DEBUG_P2P)
  printf("LIBGPU :: -- GPU-GPU Reduction  transferring result to host\n");
#endif
  
  pm->dev_set_device(0);
  
  pm->dev_pull(d_ptr[0], h_ptr, size);
  
#if defined(_DEBUG_P2P)
  printf("LIBGPU :: -- GPU-GPU Reduction  completed!\n");
#endif
}
