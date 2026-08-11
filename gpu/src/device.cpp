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

Device::Device()
{  
  pm = new PM();

  ml = new MATHLIB(pm);

  verbose_level = 0;
  
  update_dfobj = 0;
  
  rho = nullptr;
  //vj = nullptr;
  _vktmp = nullptr;

  buf_fdrv = nullptr;

  size_buf_vj = 0;
  size_buf_vk = 0;
  
  buf_vj = nullptr;
  buf_vk = nullptr;
  
  //ao2mo
  size_buf_j_pc = 0;
  size_buf_k_pc = 0;
  size_buf_ppaa = 0;
  size_buf_papa = 0;
  size_fxpp = 0;//remove when ao2mo_v3 is running
  size_bufpa = 0;//remove when ao2mo_v4 is running

  buf_j_pc = nullptr;
  buf_k_pc = nullptr;
  buf_ppaa = nullptr;
  buf_papa = nullptr;
  pin_fxpp = nullptr;//remove when ao2mo_v3 is running
  pin_bufpa = nullptr;//remove when ao2mo_v4 is running
  // h2eff_df
  size_buf_eri_h2eff = 0;
  buf_eri_h2eff = nullptr;

  // eri_impham

  size_eri_impham = 0;
  pin_eri_impham = nullptr;
  

  // tdms
  size_bravecs = 0;
  size_ketvecs = 0;
  size_dm1_full = 0;
  size_dm2_full = 0;
  h_bravecs = nullptr;
  h_ketvecs = nullptr;
  h_dm1_full = nullptr;
  h_dm2_full = nullptr;
  h_dm2_p_full = nullptr;

  // matvecs;
  size_new_sivecs=0;
  size_old_sivecs=0;
  size_ox1 = 0;
  size_instruction_list=0;
  size_op = 0;
  ox1_on_gpu = 0;
  h_new_sivecs = nullptr;
  h_old_sivecs = nullptr;
  h_ox1 = nullptr;
  h_instruction_list = nullptr;
  // The eri cache must be enabled on ALL builds (host + GPU): the _ERIS path
  // (get_dfobj_status / df_ao2mo_v4) relies on get_jk having cached blocks via
  // dd_fetch_eri. On host builds _USE_GPU is undefined, so leaving this inside
  // the #if left use_eri_cache=false and the cache was never populated.
  use_eri_cache = true;
  
  num_threads = 1;
#pragma omp parallel
  num_threads = omp_get_num_threads();

  num_devices = pm->dev_num_devices();
  
  device_data = new my_device_data[num_devices];

  for(int i=0; i<num_devices; ++i) {
    pm->dev_set_device(i);
    
    device_data[i].device_id = i;
    device_data[i].active = 0;
    
    device_data[i].jk.size_rho = 0;
    device_data[i].jk.size_vj = 0;
    device_data[i].jk.size_vk = 0;
    device_data[i].jk.size_buf1 = 0;
    device_data[i].jk.size_buf2 = 0;
    device_data[i].jk.size_buf3 = 0;
    device_data[i].jk.size_dms = 0;
    device_data[i].jk.size_dmtril = 0;
    device_data[i].jk.size_eri1 = 0;
    device_data[i].size_ucas = 0;
    device_data[i].size_umat = 0;
    device_data[i].size_h2eff = 0;
    device_data[i].size_mo_coeff = 0;
    device_data[i].size_mo_cas = 0;
    device_data[i].h2eff.size_eri_unpacked = 0; // this variable is not needed
    device_data[i].ao2mo.size_j_pc = 0;
    device_data[i].ao2mo.size_k_pc = 0;
    device_data[i].ao2mo.size_bufd = 0;
    device_data[i].ao2mo.size_bufpa = 0;
    device_data[i].ao2mo.size_bufaa = 0;
    device_data[i].h2eff.size_eri_h2eff=0;
    //pdft
    device_data[i].pdft.size_mo_grid=0;
    device_data[i].pdft.size_ao_grid=0;
    device_data[i].pdft.size_buf_pdft=0;
    device_data[i].pdft.size_cascm2=0;
    device_data[i].pdft.size_Pi=0;
    device_data[i].jk.size_rho=0;
    //fci
    device_data[i].fci.size_clinka=0;
    device_data[i].fci.size_clinkb=0;
    device_data[i].fci.size_cibra=0;
    device_data[i].fci.size_ciket=0;
    device_data[i].fci.size_tdm1=0;
    device_data[i].fci.size_tdm2=0;
    device_data[i].fci.size_tdm2_p=0;
    //matvecs
    
    
    device_data[i].jk.d_rho = nullptr;
    device_data[i].jk.d_vj = nullptr;
    device_data[i].jk.d_buf1 = nullptr;
    device_data[i].jk.d_buf2 = nullptr;
    device_data[i].jk.d_buf3 = nullptr;
    device_data[i].jk.d_vkk = nullptr;
    device_data[i].jk.d_dms = nullptr;
    device_data[i].d_mo_coeff=nullptr;
    device_data[i].d_mo_cas=nullptr;
    device_data[i].jk.d_dmtril = nullptr;
    device_data[i].jk.d_dms = nullptr;
    device_data[i].d_mo_coeff=nullptr;
    device_data[i].d_mo_cas=nullptr;
    device_data[i].jk.d_dmtril = nullptr;
    device_data[i].jk.d_eri1 = nullptr; // when not using eri cache
    device_data[i].d_ucas = nullptr;
    device_data[i].d_umat = nullptr;
    device_data[i].d_h2eff = nullptr;
    device_data[i].h2eff.d_eri_h2eff = nullptr; //for h2eff_df_v2
    
    device_data[i].fci.d_pumap_ptr = nullptr;
    
    device_data[i].ao2mo.d_j_pc = nullptr;
    device_data[i].ao2mo.d_k_pc = nullptr;
    device_data[i].ao2mo.d_bufd = nullptr;
    device_data[i].ao2mo.d_bufpa = nullptr;
    device_data[i].ao2mo.d_bufaa = nullptr;
    device_data[i].ao2mo.d_papa = nullptr;//initialized, but not allocated (used dd->jk.d_buf3)

    // pdft
    device_data[i].pdft.d_ao_grid=nullptr;
    device_data[i].pdft.d_mo_grid=nullptr;
    device_data[i].pdft.d_cascm2=nullptr;
    device_data[i].pdft.d_Pi=nullptr;
    device_data[i].pdft.d_buf_pdft1=nullptr;
    device_data[i].pdft.d_buf_pdft2=nullptr;
    //fci
    device_data[i].fci.d_clinka=nullptr;
    device_data[i].fci.d_clinkb=nullptr;
    device_data[i].fci.d_cibra=nullptr;
    device_data[i].fci.d_ciket=nullptr;
    device_data[i].fci.d_tdm1=nullptr;
    device_data[i].fci.d_tdm2=nullptr;
    device_data[i].fci.d_tdm2_p=nullptr;

#if defined (_USE_GPU)
    device_data[i].handle = nullptr;
    device_data[i].stream = nullptr;
#endif

    ml->create_handle();
  }

  t_array = (double* ) malloc(_NUM_SIMPLE_TIMER * sizeof(double));
  for(int i=0; i<_NUM_SIMPLE_TIMER; ++i) t_array[i] = 0.0;
  
  count_array = (int* ) malloc(_NUM_SIMPLE_COUNTER * sizeof(int));
  for(int i=0; i<_NUM_SIMPLE_COUNTER; ++i) count_array[i] = 0;

  // check device connectivity

  int rank = 0;
  int peer_error = pm->dev_check_peer(rank, num_devices);
  if(!peer_error) pm->dev_enable_peer(rank, num_devices);
}

/* ---------------------------------------------------------------------- */

Device::~Device()
{
  if(verbose_level) printf("LIBGPU: destroying device\n");

  pm->dev_free_host(rho);
  //pm->dev_free_host(vj);
  pm->dev_free_host(_vktmp);

  pm->dev_free_host(buf_vj);
  pm->dev_free_host(buf_vk);
  
  pm->dev_free_host(buf_fdrv);
  
  pm->dev_free_host(buf_j_pc);
  pm->dev_free_host(buf_k_pc);
  pm->dev_free_host(buf_ppaa);
  pm->dev_free_host(buf_papa);
  pm->dev_free_host(pin_fxpp);//remove 
  pm->dev_free_host(pin_bufpa);//remove when ao2mo_v3 is running

  //tdms
  pm->dev_free_host(h_bravecs);
  pm->dev_free_host(h_ketvecs);
  pm->dev_free_host(h_dm1_full);
  pm->dev_free_host(h_dm2_full);
  pm->dev_free_host(h_dm2_p_full);
  if(verbose_level) get_dev_properties(num_devices);

  if(verbose_level) { // this needs to be cleaned up and generalized...
    double total = 0.0;
    for(int i=0; i<_NUM_SIMPLE_TIMER; ++i) total += t_array[i];
  
    printf("\nLIBGPU :: SIMPLE_TIMER\n");
    printf("\nLIBGPU :: SIMPLE_TIMER :: get_jk\n");
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= init_get_jk()            time= %f s\n",0,t_array[0]);
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= pull_get_jk()            time= %f s\n",1,t_array[1]);
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= get_jk()                 time= %f s\n",2,t_array[2]);
    
    printf("\nLIBGPU :: SIMPLE_TIMER :: hessop\n");
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= hessop_get_veff()        time= %f s\n",3,t_array[3]);
    
    printf("\nLIBGPU :: SIMPLE_TIMER :: orbital_response\n");
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= orbital_response()       time= %f s\n",4,t_array[4]);
    
    
    printf("\nLIBGPU :: SIMPLE_TIMER :: _update_h2eff\n");
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= update_h2eff_sub()       time= %f s\n",5,t_array[5]);
    
    printf("\nLIBGPU :: SIMPLE_TIMER :: _h2eff_df \n");
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= h2eff_df()               time= %f s\n",6,t_array[6]);
    
    printf("\nLIBGPU :: SIMPLE_TIMER :: transfer_mo_coeff \n");
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= transfer_mo_coeff()      time= %f s\n",7,t_array[7]);
    
    printf("\nLIBGPU :: SIMPLE_TIMER :: df_ao2mo_pass1\n");
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= init_ints_and_jkpc()     time= %f s\n",8,t_array[8]);
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= compute_ints_and_jkpc()  time= %f s\n",9,t_array[9]);
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= pull_ints_and_jkpc()     time= %f s\n",10,t_array[10]);

    printf("\nLIBGPU :: SIMPLE_TIMER :: eri_impham\n");
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= init_eri_impham()     time= %f s\n",11,t_array[11]);
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= compute_eri_impham()  time= %f s\n",12,t_array[12]);
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= pull_eri_impham()     time= %f s\n",13,t_array[13]);

    printf("\nLIBGPU :: SIMPLE_TIMER :: fci_related\n");
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= init_tdm1()               time= %f s\n",14,t_array[14]);
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= init_tdm2()               time= %f s\n",15,t_array[15]);
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= push_ci()                 time= %f s\n",16,t_array[16]);
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= push_link_index()         time= %f s\n",17,t_array[17]);
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= trans_rdm1a()             time= %f s\n",18,t_array[18]);
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= trans_rdm1b()             time= %f s\n",19,t_array[19]);
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= make_rdm1a()              time= %f s\n",20,t_array[20]);
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= make_rdm1b()              time= %f s\n",21,t_array[21]);
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= tdm12kern_a()             time= %f s\n",22,t_array[22]);
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= tdm12kern_b()             time= %f s\n",23,t_array[23]);
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= tdm12kern_ab()            time= %f s\n",24,t_array[24]);
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= rdm12kern_sf()            time= %f s\n",25,t_array[25]);
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= tdm13h_spin()             time= %f s\n",26,t_array[26]);
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= tdm1h_spin()              time= %f s\n",27,t_array[27]);
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= sfudm_spin()              time= %f s\n",28,t_array[28]);
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= pptdm_spin()              time= %f s\n",29,t_array[29]);
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= pull_tdm1()               time= %f s\n",30,t_array[30]);
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= pull_tdm2()               time= %f s\n",31,t_array[31]);
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= pull_tdm13h()             time= %f s\n",32,t_array[32]);
    printf("LIBGPU :: SIMPLE_TIMER :: total= %f s\n",total);

    printf("\nLIBGPU :: SIMPLE_TIMER :: op_vecs\n");
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= push_op()            time= %f s\n",33,t_array[33]);
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= init_ox1()           time= %f s\n",34,t_array[34]);
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= init_vecs()          time= %f s\n",35,t_array[35]);
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= push_vecs()          time= %f s\n",36,t_array[36]);
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= compute_sivecs()     time= %f s\n",37,t_array[37]);
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= add_sivecs()         time= %f s\n",38,t_array[38]);
    printf("LIBGPU :: SIMPLE_TIMER :: i= %i  name= finalize_sivecs()    time= %f s\n",39,t_array[39]);

    free(t_array);
    
    
    printf("\nLIBGPU :: SIMPLE_COUNTER\n");
    printf("\nLIBGPU :: SIMPLE_COUNTER :: get_jk\n");
    printf("LIBGPU :: SIMPLE_COUNTER :: i= %i  name= get_jk()             counts= %i \n",0,count_array[0]);
    
    printf("\nLIBGPU :: SIMPLE_COUNTER :: hessop\n");
    printf("LIBGPU :: SIMPLE_COUNTER :: i= %i  name= hessop_get_veff()    counts= %i \n",1,count_array[1]);
    
    printf("\nLIBGPU :: SIMPLE_COUNTER :: orbital_response\n");
    printf("LIBGPU :: SIMPLE_COUNTER :: i= %i  name= orbital_response()   counts= %i \n",2,count_array[2]);
    
    printf("\nLIBGPU :: SIMPLE_COUNTER :: update_h2eff_sub\n");
    printf("LIBGPU :: SIMPLE_COUNTER :: i= %i  name= update_h2eff_sub()   counts= %i \n",3,count_array[3]);
    
    printf("\nLIBGPU :: SIMPLE_COUNTER :: _h2eff_df\n");
    printf("LIBGPU :: SIMPLE_COUNTER :: i= %i  name= h2eff_df()           counts= %i \n",4,count_array[4]);
    
    printf("\nLIBGPU :: SIMPLE_COUNTER :: transfer_mo_coeff\n");
    printf("LIBGPU :: SIMPLE_COUNTER :: i= %i  name= transfer_mo_coeff()  counts= %i \n",5,count_array[5]);
    
    printf("\nLIBGPU :: SIMPLE_COUNTER :: ao2mo\n");
    printf("LIBGPU :: SIMPLE_COUNTER :: i= %i  name=ao2mo_pass_v3()       counts= %i \n",6,count_array[6]);
    
    printf("\nLIBGPU :: SIMPLE_COUNTER :: eri_impham\n");
    printf("LIBGPU :: SIMPLE_COUNTER :: i= %i  name=eri_impham()          counts= %i \n",7,count_array[7]);

    
    printf("\nLIBGPU :: SIMPLE_COUNTER :: fci_kernels\n");
    printf("LIBGPU :: SIMPLE_COUNTER :: i= %i  name= trans_rdm1a()        counts= %i \n",8,count_array[8]);
    printf("LIBGPU :: SIMPLE_COUNTER :: i= %i  name= trans_rdm1b()        counts= %i \n",9,count_array[9]);
    printf("LIBGPU :: SIMPLE_COUNTER :: i= %i  name= make_rdm1a()         counts= %i \n",10,count_array[10]);
    printf("LIBGPU :: SIMPLE_COUNTER :: i= %i  name= make_rdm1b()         counts= %i \n",11,count_array[11]);
    printf("LIBGPU :: SIMPLE_COUNTER :: i= %i  name= tdm12kern_a()        counts= %i \n",12,count_array[12]);
    printf("LIBGPU :: SIMPLE_COUNTER :: i= %i  name= tdm12kern_b()        counts= %i \n",13,count_array[13]);
    printf("LIBGPU :: SIMPLE_COUNTER :: i= %i  name= tdm12kern_ab()       counts= %i \n",14,count_array[14]);
    printf("LIBGPU :: SIMPLE_COUNTER :: i= %i  name= rdm12kern_sf()       counts= %i \n",15,count_array[15]);
    printf("LIBGPU :: SIMPLE_COUNTER :: i= %i  name= tdm13h_spin()        counts= %i \n",16,count_array[16]);
    printf("LIBGPU :: SIMPLE_COUNTER :: i= %i  name= tdm1h_spin()         counts= %i \n",17,count_array[17]);
    printf("LIBGPU :: SIMPLE_COUNTER :: i= %i  name= sfudm_spin()         counts= %i \n",18,count_array[18]);
    printf("LIBGPU :: SIMPLE_COUNTER :: i= %i  name= ppdm_spin()          counts= %i \n",19,count_array[19]);
    printf("LIBGPU :: SIMPLE_COUNTER :: i= %i  name= pull_tdm1()          counts= %i \n",20,count_array[20]);
    printf("LIBGPU :: SIMPLE_COUNTER :: i= %i  name= pull_tdm2()          counts= %i \n",21,count_array[21]);
    printf("LIBGPU :: SIMPLE_COUNTER :: i= %i  name= pull_tdm13h()        counts= %i \n",22,count_array[22]);

    printf("\nLIBGPU :: SIMPLE_COUNTER :: op_vec kernels\n");
    printf("LIBGPU :: SIMPLE_COUNTER :: i= %i  name= push_op()            counts= %i \n",23,count_array[23]);
    printf("LIBGPU :: SIMPLE_COUNTER :: i= %i  name= init_ox1()           counts= %i \n",24,count_array[24]);
    printf("LIBGPU :: SIMPLE_COUNTER :: i= %i  name= init_vecs()          counts= %i \n",25,count_array[25]);
    printf("LIBGPU :: SIMPLE_COUNTER :: i= %i  name= push_vecs()          counts= %i \n",26,count_array[26]);
    printf("LIBGPU :: SIMPLE_COUNTER :: i= %i  name= compute_sivecs()     counts= %i \n",27,count_array[27]);
    printf("LIBGPU :: SIMPLE_COUNTER :: i= %i  name= add_sivecs()         counts= %i \n",28,count_array[28]);
    printf("LIBGPU :: SIMPLE_COUNTER :: i= %i  name= finalize_sivecs()    counts= %i \n",29,count_array[29]);
 
    free(count_array);
  }

  // print summary of cached eri blocks

  if(use_eri_cache) {
    if(verbose_level) {
      printf("\nLIBGPU :: eri cache statistics :: count= %zu\n",eri_list.size());
      for(int i=0; i<eri_list.size(); ++i)
	printf("LIBGPU :: %i : eri= %p  Mbytes= %f  count= %i  update= %i device= %i\n", i, (void*) eri_list[i],
	       eri_size[i]*sizeof(double)/1024./1024., eri_count[i], eri_update[i], eri_device[i]);
    }
    
    eri_count.clear();
    eri_size.clear();
#ifdef _DEBUG_ERI_CACHE
    for(int i=0; i<d_eri_host.size(); ++i) pm->dev_free_host( d_eri_host[i] );
#endif
    for(int i=0; i<d_eri_cache.size(); ++i) {
      int id = eri_device[i];
      pm->dev_set_device(id);
      pm->dev_free(d_eri_cache[i], "eri_cache");
    }
    eri_list.clear();
  }
  
  for(int i=0; i<num_devices; ++i) {
  
    pm->dev_set_device(i);
    
    my_device_data * dd = &(device_data[i]);
    
    pm->dev_free(dd->jk.d_rho, "rho");
    pm->dev_free(dd->jk.d_vj, "vj");
    pm->dev_free(dd->jk.d_buf1, "buf1");
    pm->dev_free(dd->jk.d_buf2, "buf2");
    pm->dev_free(dd->jk.d_buf3, "buf3");
    pm->dev_free(dd->jk.d_vkk, "vkk");
    pm->dev_free(dd->jk.d_dms, "dms");
    pm->dev_free(dd->d_mo_coeff, "mo_coeff");
    pm->dev_free(dd->d_mo_cas, "mo_cas");
    pm->dev_free(dd->jk.d_dmtril, "dmtril");
    pm->dev_free(dd->jk.d_eri1, "eri1");
    pm->dev_free(dd->d_ucas, "ucas");
    pm->dev_free(dd->d_umat, "umat");
    pm->dev_free(dd->d_h2eff, "h2eff");
    pm->dev_free(dd->h2eff.d_eri_h2eff, "eri_h2eff");
    
    pm->dev_free(dd->ao2mo.d_j_pc, "j_pc");
    pm->dev_free(dd->ao2mo.d_k_pc, "k_pc");

    pm->dev_free(dd->pdft.d_ao_grid, "ao_grid");
    pm->dev_free(dd->pdft.d_mo_grid, "ao_grid");
    pm->dev_free(dd->pdft.d_cascm2, "cascm2");
    pm->dev_free(dd->pdft.d_Pi, "Pi");
    pm->dev_free(dd->pdft.d_buf_pdft1, "buf_pdft1");
    pm->dev_free(dd->pdft.d_buf_pdft2, "buf_pdft2");

    pm->dev_free(dd->ao2mo.d_bufpa, "bufpa");
    pm->dev_free(dd->ao2mo.d_bufd, "bufd");
    pm->dev_free(dd->ao2mo.d_bufaa, "bufaa");

    pm->dev_free(dd->fci.d_clinka, "clinka");
    pm->dev_free(dd->fci.d_clinkb, "clinkb");
    pm->dev_free(dd->fci.d_cibra, "cibra");
    pm->dev_free(dd->fci.d_ciket, "ciket");
    pm->dev_free(dd->fci.d_tdm1, "tdm1");
    pm->dev_free(dd->fci.d_tdm2, "tdm2");
    pm->dev_free(dd->fci.d_tdm2_p, "tdm2_p");


    for(int i=0; i<dd->fci.size_pumap.size(); ++i) {
      pm->dev_free_host(dd->fci.pumap[i]);
      
      std::string name = "pumap-" + std::to_string(i);
      pm->dev_free(dd->fci.d_pumap[i], name);
    }
    dd->fci.type_pumap.clear();
    dd->fci.size_pumap.clear();
    dd->fci.pumap.clear();
    dd->fci.d_pumap.clear();
  }

  if(verbose_level) {
    pm->print_mem_summary();
    
    printf("LIBGPU :: Finished\n");
  }

  delete [] device_data;
  
  delete ml;
  
  delete pm;
}

/* ---------------------------------------------------------------------- */

// xthi.c from http://docs.cray.com/books/S-2496-4101/html-S-2496-4101/cnlexamples.html

#if !defined(__linux__)
#define CPU_SETSIZE 1024
#endif

// util-linux-2.13-pre7/schedutils/taskset.c
void Device::get_cores(char *str)
{
#if defined(__linux__)
  cpu_set_t mask;
  sched_getaffinity(0, sizeof(cpu_set_t), &mask);

  char *ptr = str;
  int i, j, entry_made = 0;
  for (i = 0; i < CPU_SETSIZE; i++) {
    if (CPU_ISSET(i, &mask)) {
      int run = 0;
      entry_made = 1;
      for (j = i + 1; j < CPU_SETSIZE; j++) {
        if (CPU_ISSET(j, &mask)) run++;
        else break;
      }
      if (!run)
        sprintf(ptr, "%d,", i);
      else if (run == 1) {
        sprintf(ptr, "%d,%d,", i, i + 1);
        i++;
      } else {
        sprintf(ptr, "%d-%d,", i, i + run);
        i += run;
      }
      while (*ptr != 0) ptr++;
    }
  }
  ptr -= entry_made;
  *ptr = 0;
#else
  str[0] = 0;
#endif
}

/* ---------------------------------------------------------------------- */

int Device::get_num_devices()
{
  if(verbose_level) printf("LIBGPU: getting number of devices\n");
  return pm->dev_num_devices();
}

/* ---------------------------------------------------------------------- */
    
void Device::get_dev_properties(int N)
{
  printf("LIBGPU: reporting device properties N= %i\n",N);
  
  char nname[16];
  gethostname(nname, 16);
  int rnk = 0;
  
#pragma omp parallel for ordered
  for(int it=0; it<num_threads; ++it) {
    char list_cores[7*CPU_SETSIZE];
    get_cores(list_cores);
#pragma omp ordered
    printf("LIBGPU: To affinity and beyond!! nname= %s  rnk= %d  tid= %d: list_cores= (%s)\n",
	   nname, rnk, omp_get_thread_num(), list_cores);
  }
  
  pm->dev_properties(N);
}

/* ---------------------------------------------------------------------- */
    
void Device::set_device(int id)
{
  if(verbose_level) printf("LIBGPU: setting device id= %i\n",id);
  pm->dev_set_device(id);
}

/* ---------------------------------------------------------------------- */
    
void Device::barrier_all()
{
  //if(verbose_level) printf("LIBGPU: barrier on all devices\n");

  for(int i=0; i<num_devices; ++i) {
    pm->dev_set_device(i);
    pm->dev_barrier();
  }
}

/* ---------------------------------------------------------------------- */
    
void Device::set_update_dfobj_(int _val)
{
  update_dfobj = _val; // this is reset to zero in Device::pull_get_jk
}

/* ---------------------------------------------------------------------- */
    
void Device::disable_eri_cache_()
{
  use_eri_cache = false;
  printf("LIBGPU :: Error : Not able to disable eri caching as additional support needs to be added to track eri_extra array.");
  exit(1);
}

/* ---------------------------------------------------------------------- */
    
void Device::set_verbose_(int _verbose)
{
  verbose_level = _verbose; // setting nonzero prints affinity + timing info
}

/* ---------------------------------------------------------------------- */

// return stored values for Python side to make decisions
// update_dfobj == true :: nothing useful to return if need to update eri blocks on device
// count_ == -1 :: return # of blocks cached for dfobj
// count_ >= 0 :: return extra data for cached block

void Device::get_dfobj_status(size_t addr_dfobj, py::array_t<int> _arg)
{
  py::buffer_info info_arg = _arg.request();
  int * arg = static_cast<int*>(info_arg.ptr);
  
  int naux_ = arg[0];
  int nao_pair_ = arg[1];
  int count_ = arg[2];
  int update_dfobj_ = arg[3];
  
  // printf("Inside get_dfobj_status(): addr_dfobj= %#012x  naux_= %i  nao_pair_= %i  count_= %i  update_dfobj_= %i\n",
  // 	 addr_dfobj, naux_, nao_pair_, count_, update_dfobj_);
  
  update_dfobj_ = update_dfobj;

  // nothing useful to return if need to update eri blocks on device
  
  if(update_dfobj) { 
    // printf("Leaving get_dfobj_status(): addr_dfobj= %#012x  update_dfobj_= %i\n", addr_dfobj, update_dfobj_);
    
    arg[3] = update_dfobj_;
    return;
  }
  
  // return # of blocks cached for dfobj

  if(count_ == -1) {
    int id = eri_list.size();
    for(int i=0; i<eri_list.size(); ++i)
      if(eri_list[i] == addr_dfobj) {
	id = i;
	break;
      }

    if(id < eri_list.size()) count_ = eri_num_blocks[id];
    
    // printf("Leaving get_dfobj_status(): addr_dfobj= %#012x  count_= %i  update_dfobj_= %i\n", addr_dfobj, count_, update_dfobj_);

    arg[2] = count_;
    arg[3] = update_dfobj_;
    return;
  }

  // return extra data for cached block
  
  int id = eri_list.size();
  for(int i=0; i<eri_list.size(); ++i)
    if(eri_list[i] == addr_dfobj+count_) {
      id = i;
      break;
    }

  // printf("eri_list.size()= %i  id= %i\n",eri_list.size(), id);
  
  naux_ = -1;
  nao_pair_ = -1;
  
  if(id < eri_list.size()) {
  
    naux_     = eri_extra[id * _ERI_CACHE_EXTRA    ];
    nao_pair_ = eri_extra[id * _ERI_CACHE_EXTRA + 1];

  }

  arg[0] = naux_;
  arg[1] = nao_pair_;
  arg[2] = count_;
  arg[3] = update_dfobj_;
  
  // printf("Leaving get_dfobj_status(): addr_dfobj= %#012x  id= %i  naux_= %i  nao_pair_= %i  count_= %i  update_dfobj_= %i\n",
  // 	 addr_dfobj, id, naux_, nao_pair_, count_, update_dfobj_);
  
  // printf("Leaving get_dfobj_status(): addr_dfobj= %#012x  id= %i  arg= %i %i %i %i\n",
  // 	 addr_dfobj, id, arg[0], arg[1], arg[2], arg[3]);
}

/* ---------------------------------------------------------------------- */


/* ---------------------------------------------------------------------- */

void Device::push_mo_coeff(py::array_t<double> _mo_coeff, int _size_mo_coeff)
{
  double t0 = omp_get_wtime();
  
  py::buffer_info info_mo_coeff = _mo_coeff.request(); // 2D array (naux, nao_pair)

  double * mo_coeff = static_cast<double*>(info_mo_coeff.ptr);

  // host pushes to each device; optimize later host->device0 plus device-device transfers (i.e. bcast)

#if defined(_ENABLE_P2P)
  std::vector<double *> mo_vec(num_devices); // array of device addresses 
    
  for(int id=0; id<num_devices; ++id) {
    pm->dev_set_device(id);
    
    my_device_data * dd = &(device_data[id]);

    grow_array(dd->d_mo_coeff, _size_mo_coeff, dd->size_mo_coeff, "mo_coeff", FLERR);
    
    mo_vec[id] = dd->d_mo_coeff;
  }
    
  mgpu_bcast(mo_vec, mo_coeff, _size_mo_coeff*sizeof(double)); // host -> gpu 0, then Bcast to all gpu

#else
  for(int id=0; id<num_devices; ++id) {
    
    pm->dev_set_device(id);
  
    my_device_data * dd = &(device_data[id]);
    
    grow_array(dd->d_mo_coeff, _size_mo_coeff, dd->size_mo_coeff, "mo_coeff", FLERR);
    
    pm->dev_push_async(dd->d_mo_coeff, mo_coeff, _size_mo_coeff*sizeof(double));
  }
#endif
  
  double t1 = omp_get_wtime();
  t_array[7] += t1 - t0;
  count_array[5] +=1;
}

/* ---------------------------------------------------------------------- */


/* ---------------------------------------------------------------------- */

// Is both _ocm2 in/out as it get over-written and resized?

void Device::orbital_response(py::array_t<double> _f1_prime,
			      py::array_t<double> _ppaa, py::array_t<double> _papa, py::array_t<double> _eri_paaa,
			      py::array_t<double> _ocm2, py::array_t<double> _tcm2, py::array_t<double> _gorb,
			      int ncore, int nocc, int nmo) // obselete
{
  double t0 = omp_get_wtime();
    
  py::buffer_info info_ppaa = _ppaa.request(); // 4D array (26, 26, 2, 2)
  py::buffer_info info_papa = _papa.request(); // 4D array (26, 2, 26, 2)
  py::buffer_info info_paaa = _eri_paaa.request();
  py::buffer_info info_ocm2 = _ocm2.request();
  py::buffer_info info_tcm2 = _tcm2.request();
  py::buffer_info info_gorb = _gorb.request();
  
  double * ppaa = static_cast<double*>(info_ppaa.ptr);
  double * papa = static_cast<double*>(info_papa.ptr);
  double * paaa = static_cast<double*>(info_paaa.ptr);
  double * ocm2 = static_cast<double*>(info_ocm2.ptr);
  double * tcm2 = static_cast<double*>(info_tcm2.ptr);
  double * gorb = static_cast<double*>(info_gorb.ptr);
  
  int ocm2_size_2d = info_ocm2.shape[2] * info_ocm2.shape[3];
  int ocm2_size_3d = info_ocm2.shape[1] * ocm2_size_2d;
    
  // printf("LIBGPU: Inside libgpu_orbital_response()\n");
  // printf("  -- ncore= %i\n",ncore); // 7
  // printf("  -- nocc=  %i\n",nocc); // 9
  // printf("  -- nmo=   %i\n",nmo); // 26
  // printf("  -- ppaa: ndim= %i\n",info_ppaa.ndim);
  // printf("  --       shape=");
  // for(int i=0; i<info_ppaa.ndim; ++i) printf(" %i", info_ppaa.shape[i]);
  // printf("\n");

  double * f1_prime = (double *) pm->dev_malloc_host(nmo*nmo*sizeof(double));
  
  // loop over f1 (i<nmo)
#pragma omp parallel for
  for(int p=0; p<nmo; ++p) {
    
    double * f1 = &(f1_prime[p * nmo]);
  
    // pointers to slices of data
    
    int ppaa_size_2d = info_ppaa.shape[2] * info_ppaa.shape[3];
    int ppaa_size_3d = info_ppaa.shape[1] * ppaa_size_2d;
    double * praa = &(ppaa[p * ppaa_size_3d]); // ppaa.shape[1] x ppaa.shape[2] x ppaa.shape[3]
    
    int papa_size_2d = info_papa.shape[2] * info_papa.shape[3];
    int papa_size_3d = info_papa.shape[1] * papa_size_2d;
    double * para = &(papa[p * papa_size_3d]); // papa.shape[1] x papa.shape[2] x papa.shape[3]
    
    double * paaa = &(ppaa[p * ppaa_size_3d + ncore * ppaa_size_2d]); // (nocc-ncore) x ppaa.shape[2] x ppaa.shape[3]
    
    // ====================================================================
    // iteration (0, ncore)
    // ====================================================================
    
    // construct ra, ar, cm
    
    double * ra = praa; // (ncore,2,2)
    
    //int indx = 0;
   
    double * cm = ocm2; // (2, 2, 2, ncore)
    
    for(int i=0; i<nmo; ++i) f1[i] = 0.0;
    
    // tensordot(paaa, cm, axes=((0,1,2), (2,1,0)))
    
    // printf("f1 += paaa{%i, %i, %i} X cm{%i, %i, %i, %i}\n",
    // 	   nocc-ncore,info_ppaa.shape[2],info_ppaa.shape[3],
    // 	   info_ocm2.shape[0],info_ocm2.shape[1],info_ocm2.shape[2],ncore);
    
    for(int i=0; i<ncore; ++i) {
      
      double val = 0.0;
      for(int k1=0; k1<info_ppaa.shape[3]; ++k1)
	for(int j1=0; j1<info_ppaa.shape[2]; ++j1)
	  for(int i1=0; i1<nocc-ncore; ++i1)
	    {
	      int indx1 = i1 * ppaa_size_2d + j1 * info_ppaa.shape[3] + k1;
	      int indx2 = k1 * ocm2_size_3d + j1 * ocm2_size_2d + i1 * info_ocm2.shape[3] + i;
	      val += paaa[indx1] * cm[indx2];
	    }
      
      f1[i] += val;
    }
    
    // tensordot(ra, cm, axes=((0,1,2), (3,0,1)))
    
    // printf("f1 += ra{%i, %i, %i} X cm{%i, %i, %i, %i}\n",
    // 	   ncore,info_ppaa.shape[2],info_ppaa.shape[3],
    // 	   info_ocm2.shape[0],info_ocm2.shape[1],info_ocm2.shape[2],ncore);
    
    for(int i=0; i<info_ocm2.shape[2]; ++i) {
      
      double val = 0.0;
      for(int k1=0; k1<info_ppaa.shape[3]; ++k1)
	for(int j1=0; j1<info_ppaa.shape[2]; ++j1)
	  for(int i1=0; i1<ncore; ++i1)
	    {
	      int indx1 = i1 * ppaa_size_2d + j1 * info_ppaa.shape[3] + k1;
	      int indx2 = j1 * ocm2_size_3d + k1 * ocm2_size_2d + i * info_ocm2.shape[3] + i1;
	      val += ra[indx1] * cm[indx2];
	    }
      
      f1[ncore+i] += val;
    }
    
    // tensordot(ar, cm, axes=((0,1,2), (0,3,2)))
    
    // printf("f1 += ar{%i, %i, %i} X cm{%i, %i, %i, %i}\n",
    // 	   info_papa.shape[1], ncore, info_papa.shape[3],
    // 	   info_ocm2.shape[0],info_ocm2.shape[1],info_ocm2.shape[2],ncore);

    for(int i=0; i<info_ocm2.shape[1]; ++i) {
      
      double val = 0.0;
      for(int k1=0; k1<info_ppaa.shape[3]; ++k1)
	for(int j1=0; j1<ncore; ++j1)
	  for(int i1=0; i1<info_papa.shape[1]; ++i1)
	    {
	      int indx1 = i1 * papa_size_2d + j1 * info_papa.shape[3] + k1;
	      int indx2 = i1 * ocm2_size_3d + i * ocm2_size_2d + k1 * info_ocm2.shape[3] + j1;
	      val += para[indx1] * cm[indx2];
	    }
      
      f1[ncore+i] += val;
    }
    
    // tensordot(ar, cm, axes=((0,1,2), (1,3,2)))
    
    // printf("f1 += ar{%i, %i, %i} X cm{%i, %i, %i, %i}\n",
    // 	   info_papa.shape[1], ncore, info_papa.shape[3],
    // 	   info_ocm2.shape[0],info_ocm2.shape[1],info_ocm2.shape[2],ncore);

    for(int i=0; i<info_ocm2.shape[0]; ++i) {
      
      double val = 0.0;
      for(int k1=0; k1<info_ppaa.shape[3]; ++k1)
	for(int j1=0; j1<ncore; ++j1)
	  for(int i1=0; i1<info_papa.shape[1]; ++i1)
	    {
	      int indx1 = i1 * papa_size_2d + j1 * info_papa.shape[3] + k1;
	      int indx2 = i * ocm2_size_3d + i1 * ocm2_size_2d + k1 * info_ocm2.shape[3] + j1;
	      val += para[indx1] * cm[indx2];
	    }
      
      f1[ncore+i] += val;
    }
    
    // ====================================================================
    // iteration (nocc, nmo)
    // ====================================================================
    
    // paaa = praa[ncore:nocc, :, :] = ppaa[p, ncore:nocc, :, :]
    // ra = praa[i:j] = ppaa[p, nocc:nmo, :, :]
    // ar = para[:, i:j] = papa[p, :, nocc:nmo, :]
    // cm = ocm2[:, :, :, i:j] = ocm2[:, :, :, nocc:nmo]

    // tensordot(paaa, cm, axes=((0,1,2), (2,1,0)))
    
    // printf("f1 += paaa{%i, %i, %i} X cm{%i, %i, %i, %i}\n",
    // 	   nmo-nocc,info_ppaa.shape[2],info_ppaa.shape[3],
    // 	   info_ocm2.shape[0],info_ocm2.shape[1],info_ocm2.shape[2],nmo-nocc);
    
    for(int i=nocc; i<nmo; ++i) {
      
      double val = 0.0;
      //int indx = 0;
      for(int k1=0; k1<info_ppaa.shape[3]; ++k1)
	for(int j1=0; j1<info_ppaa.shape[2]; ++j1)
	  for(int i1=0; i1<nocc-ncore; ++i1)
	    {
	      int indx1 = i1 * ppaa_size_2d + j1 * info_ppaa.shape[3] + k1;
	      int indx2 = k1 * ocm2_size_3d + j1 * ocm2_size_2d + i1 * info_ocm2.shape[3] + i;
	      val += paaa[indx1] * cm[indx2];
	    }
      
      f1[i] += val;
    }
    
    // tensordot(ra, cm, axes=((0,1,2), (3,0,1)))
    
    // printf("f1 += ra{%i, %i, %i} X cm{%i, %i, %i, %i}\n",
    // 	   nmo-nocc,info_ppaa.shape[2],info_ppaa.shape[3],
    // 	   info_ocm2.shape[0],info_ocm2.shape[1],info_ocm2.shape[2],nmo-nocc);
    
    for(int i=0; i<info_ocm2.shape[2]; ++i) {
      
      double val = 0.0;
      for(int k1=0; k1<info_ppaa.shape[3]; ++k1)
	for(int j1=0; j1<info_ppaa.shape[2]; ++j1)
	  for(int i1=0; i1<nmo-nocc; ++i1)
	    {
	      int indx1 = (nocc+i1) * ppaa_size_2d + j1 * info_ppaa.shape[3] + k1;
	      int indx2 = j1 * ocm2_size_3d + k1 * ocm2_size_2d + i * info_ocm2.shape[3] + (nocc+i1);
	      val += ra[indx1] * cm[indx2];
	    }
      
      f1[ncore+i] += val;
    }
    
    // tensordot(ar, cm, axes=((0,1,2), (0,3,2)))
    
    // printf("f1 += ar{%i, %i, %i} X cm{%i, %i, %i, %i}\n",
    // 	   info_papa.shape[1], nmo-nocc, info_papa.shape[3],
    // 	   info_ocm2.shape[0],info_ocm2.shape[1],info_ocm2.shape[2],nmo-nocc);

    for(int i=0; i<info_ocm2.shape[1]; ++i) {
      
      double val = 0.0;
      for(int k1=0; k1<info_ppaa.shape[3]; ++k1)
	for(int j1=0; j1<nmo-nocc; ++j1)
	  for(int i1=0; i1<info_papa.shape[1]; ++i1)
	    {
	      int indx1 = i1 * papa_size_2d + (nocc+j1) * info_papa.shape[3] + k1;
	      int indx2 = i1 * ocm2_size_3d + i * ocm2_size_2d + k1 * info_ocm2.shape[3] + (nocc+j1);
	      val += para[indx1] * cm[indx2];
	    }
      
      f1[ncore+i] += val;
    }
    
    // tensordot(ar, cm, axes=((0,1,2), (1,3,2)))
    
    // printf("f1 += ar{%i, %i, %i} X cm{%i, %i, %i, %i}\n",
    // 	   info_papa.shape[1], nmo-nocc, info_papa.shape[3],
    // 	   info_ocm2.shape[0],info_ocm2.shape[1],info_ocm2.shape[2],nmo-nocc);

    for(int i=0; i<info_ocm2.shape[0]; ++i) {
      
      double val = 0.0;
      for(int k1=0; k1<info_ppaa.shape[3]; ++k1)
	for(int j1=0; j1<nmo-nocc; ++j1)
	  for(int i1=0; i1<info_papa.shape[1]; ++i1)
	    {
	      int indx1 = i1 * papa_size_2d + (nocc+j1) * info_papa.shape[3] + k1;
	      int indx2 = i * ocm2_size_3d + i1 * ocm2_size_2d + k1 * info_ocm2.shape[3] + (nocc+j1);
	      val += para[indx1] * cm[indx2];
	    }
      
      f1[ncore+i] += val;
    }
  } // for(p<nmo)

  // # (H.x_aa)_va, (H.x_aa)_ac

  int _ocm2_size_1d = nocc - ncore;
  int _ocm2_size_2d = info_ocm2.shape[2] * _ocm2_size_1d;
  int _ocm2_size_3d = info_ocm2.shape[1] * ocm2_size_2d;
  int size_ecm = info_ocm2.shape[0] * info_ocm2.shape[1] * info_ocm2.shape[2] * (nocc-ncore);
  
  double * _ocm2t = (double *) pm->dev_malloc_host(size_ecm * sizeof(double));
  double * ecm2 = (double *) pm->dev_malloc_host(size_ecm * sizeof(double)); // tmp space and ecm2
  
  // ocm2 = ocm2[:,:,:,ncore:nocc] + ocm2[:,:,:,ncore:nocc].transpose (1,0,3,2)

  int indx = 0;
  double * _ocm2_tmp = ecm2;
  for(int i=0; i<info_ocm2.shape[0]; ++i)
    for(int j=0; j<info_ocm2.shape[1]; ++j)
      for(int k=0; k<info_ocm2.shape[2]; ++k)
        for(int l=0; l<(nocc-ncore); ++l)
          {
            int indx1 = i * ocm2_size_3d + j * ocm2_size_2d + k * info_ocm2.shape[3] + (ncore+l);
            int indx2 = j * ocm2_size_3d + i * ocm2_size_2d + l * info_ocm2.shape[3] + (ncore+k);
            _ocm2_tmp[indx++] = ocm2[indx1] + ocm2[indx2];
          }

  // ocm2 += ocm2.transpose (2,3,0,1)

  _ocm2_size_3d = info_ocm2.shape[1] * _ocm2_size_2d;
  
  indx = 0;
  for(int i=0; i<info_ocm2.shape[0]; ++i)
    for(int j=0; j<info_ocm2.shape[1]; ++j)
      for(int k=0; k<info_ocm2.shape[2]; ++k)
       for(int l=0; l<(nocc-ncore); ++l)
	 {
	   int indx1 = i * _ocm2_size_3d + j * _ocm2_size_2d + k * _ocm2_size_1d + l;
	   int indx2 = k * _ocm2_size_3d + l * _ocm2_size_2d + i * _ocm2_size_1d + j;
	   _ocm2t[indx] = _ocm2_tmp[indx1] + _ocm2_tmp[indx2];
	   indx++;
	 }
    
  // ecm2 = ocm2 + tcm2
  
  for(int i=0; i<size_ecm; ++i) ecm2[i] = _ocm2t[i] + tcm2[i];
  
  // f1_prime[:ncore,ncore:nocc] += np.tensordot (self.eri_paaa[:ncore], ecm2, axes=((1,2,3),(1,2,3)))
  
  int paaa_size_1d = info_paaa.shape[3];
  int paaa_size_2d = info_paaa.shape[2] * paaa_size_1d;
  int paaa_size_3d = info_paaa.shape[1] * paaa_size_2d;
  
  for(int i=0; i<ncore; ++i) 
    for(int j=0; j<(nocc-ncore); ++j) {
      
      double val = 0.0;
      for(int k1=0; k1<info_ppaa.shape[3]; ++k1)
	for(int j1=0; j1<info_paaa.shape[2]; ++j1)
	  for(int i1=0; i1<info_paaa.shape[1]; ++i1)
	    {
	      int indx1 = i * paaa_size_3d + i1 * paaa_size_2d + j1 * paaa_size_1d + k1;
	      int indx2 = j * _ocm2_size_3d + i1 * _ocm2_size_2d + j1 * _ocm2_size_1d + k1;
	      val += paaa[indx1] * ecm2[indx2];
	    }
      
      f1_prime[i*nmo+ncore+j] += val;
    }
    
  // f1_prime[nocc:,ncore:nocc] += np.tensordot (self.eri_paaa[nocc:], ecm2, axes=((1,2,3),(1,2,3)))

  for(int i=nocc; i<nmo; ++i) 
    for(int j=0; j<(nocc-ncore); ++j) {
      
      double val = 0.0;
      for(int k1=0; k1<info_ppaa.shape[3]; ++k1)
	for(int j1=0; j1<info_paaa.shape[2]; ++j1)
	  for(int i1=0; i1<info_paaa.shape[1]; ++i1)
	    {
	      int indx1 = i * paaa_size_3d + i1 * paaa_size_2d + j1 * paaa_size_1d + k1;
	      int indx2 = j * _ocm2_size_3d + i1 * _ocm2_size_2d + j1 * _ocm2_size_1d + k1;
	      val += paaa[indx1] * ecm2[indx2];
	    }
      
      f1_prime[i*nmo+ncore+j] += val;
    }
  
  // return gorb + (f1_prime - f1_prime.T)

  double * g_f1_prime = (double *) pm->dev_malloc_host(nmo*nmo*sizeof(double));
  
  indx = 0;
  for(int i=0; i<nmo; ++i)
    for(int j=0; j<nmo; ++j) {
      int indx1 = i * nmo + j;
      int indx2 = j * nmo + i;
      g_f1_prime[indx] = gorb[indx] + f1_prime[indx1] - f1_prime[indx2];
      indx++;
    }
  
  py::buffer_info info_f1_prime = _f1_prime.request();
  double * res = static_cast<double*>(info_f1_prime.ptr);

  for(int i=0; i<nmo*nmo; ++i) res[i] = g_f1_prime[i];

  double t1 = omp_get_wtime();
  t_array[4]  += t1  - t0;
  count_array[2] += 1; 
  
#if 0
  pm->dev_free_host(ar_global);
#endif
  
  pm->dev_free_host(g_f1_prime);
  pm->dev_free_host(ecm2);
  pm->dev_free_host(_ocm2t);
  pm->dev_free_host(f1_prime);

}

/* ---------------------------------------------------------------------- */
