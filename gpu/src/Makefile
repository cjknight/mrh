
# Definition of MACROS

BINROOT=./
EXE=libgpu.so
SHELL=/bin/sh
CXX = g++
CXXFLAGS=
FC=gfortran
FCFLAGS=
LD = $(CXX)
LDFLAGS = -fPIC -shared
AR = ar rcs
CPP = cpp -P -traditional
INSTALL=../

CUDA_CXX = $(CXX)
CUDA_CXXFLAGS = 

ARCH ?= polaris-gnu-nvcc
include arch/$(ARCH)

# couple more flags to build shared library (allows arch files to be used for general app building)
CXXFLAGS += -fPIC
ifneq ($(findstring _USE_CPU,$(CXXFLAGS)),_USE_CPU)
CUDA_CXXFLAGS += -shared -Xcompiler -fPIC
endif
LDFLAGS += -fPIC -shared

# -- subset of src files with cuda kernels (need nvcc compiler)
CUDA_SRC = pm/cuda/pm_cuda.cpp pm/cuda/jk.cpp pm/cuda/ao2mo.cpp pm/cuda/h2eff.cpp pm/cuda/impham.cpp pm/cuda/pdft.cpp pm/cuda/fci.cpp pm/cuda/utilities.cpp
CUDA_INC = $(CUDA_SRC:.cpp=.h)
TMP0 = $(CUDA_SRC:pm/%=%)
CUDA_OBJ = $(TMP0:.cpp=.o)

# -- subset of src files in src/device directory
DEV_SRC = $(wildcard device/*.cpp)
DEV_INC = $(wildcard device/*.h)
TMPD = $(DEV_SRC:device/%=%)
DEV_OBJ = $(TMPD:.cpp=.o)

# -- subset of src files in src directory
C_SRC = $(wildcard *.cpp)
C_INC = $(wildcard *.h)
C_OBJ = $(C_SRC:.cpp=.o)

# -- subset of src files in src/pm directory (and per-platform subdirs)
PM_SRC = $(filter-out $(CUDA_SRC), $(wildcard pm/*.cpp) $(wildcard pm/sycl/*.cpp) $(wildcard pm/hip/*.cpp) $(wildcard pm/openmp/*.cpp) $(wildcard pm/host/*.cpp))
PM_INC = $(filter-out $(CUDA_INC), $(wildcard pm/*.h) $(wildcard pm/sycl/*.h) $(wildcard pm/hip/*.h) $(wildcard pm/openmp/*.h) $(wildcard pm/host/*.h))
TMP1 = $(PM_SRC:pm/%=%)
PM_OBJ = $(TMP1:.cpp=.o)

# -- subset of src files in src/mathlib directory
ML_SRC = $(wildcard mathlib/*.cpp)
ML_INC = $(wildcard mathlib/*.h)
TMP2 = $(ML_SRC:mathlib/%=%)
ML_OBJ = $(TMP2:.cpp=.o)

#$(info CUDA_SRC is [${CUDA_SRC}])
#$(info CUDA_INC is [${CUDA_INC}])
#$(info CUDA_OBJ is [${CUDA_OBJ}])

#$(info C_SRC is [${C_SRC}])
#$(info C_INC is [${C_INC}])
#$(info C_OBJ is [${C_OBJ}])

#$(info PM_SRC is [${PM_SRC}])
#$(info PM_INC is [${PM_INC}])
#$(info PM_OBJ is [${PM_OBJ}])

#$(info ML_SRC is [${ML_SRC}])
#$(info ML_INC is [${ML_INC}])
#$(info ML_OBJ is [${ML_OBJ}])

# -- target : 	Dependencies
# --		Rule to create target

$(EXE): $(C_OBJ) $(DEV_OBJ) $(CUDA_OBJ) $(PM_OBJ) $(ML_OBJ)
	$(LD) $(LDFLAGS) -o $@ $(C_OBJ) $(DEV_OBJ) $(CUDA_OBJ) $(PM_OBJ) $(ML_OBJ) $(LIB)

install: $(EXE)
	 cp $(EXE) $(INSTALL)

####################################################################

$(C_OBJ): %.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<

$(DEV_OBJ): %.o: device/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(PM_OBJ): %.o: pm/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(ML_OBJ): %.o: mathlib/%.cpp
	$(CXX) $(CXXFLAGS) -c $<

$(CUDA_OBJ): %.o: pm/%.cpp
	@mkdir -p $(dir $@)
	$(CUDA_CXX) $(CUDA_CXXFLAGS) -c $< -o $@

#
# -- Remove *.o and *~ from the directory
clean:
	rm -f *.o *~ ./$(EXE)
	rm -f device/*.o cuda/*.o sycl/*.o hip/*.o openmp/*.o host/*.o
	rm -f $(INSTALL)/$(EXE)
	rm -rf $(EXE).dSYM
#
# -- Remove *.o, *~, and executable from the directory
realclean:
	rm -f *.o *~ ./$(EXE)
	rm -f device/*.o cuda/*.o sycl/*.o hip/*.o openmp/*.o host/*.o
	rm -f $(INSTALL)/$(EXE)
	rm -rf $(EXE).dSYM
	rm -f *.optrpt

#
# -- Simple dependencies

libgpu.o : libgpu.cpp libgpu.h device/device.h

cuda/pm_cuda.o : pm/cuda/pm_cuda.cpp pm/cuda/pm_cuda.h pm/pm.h
sycl/pm_sycl.o : pm/sycl/pm_sycl.cpp pm/sycl/pm_sycl.h pm/pm.h
hip/pm_hip.o : pm/hip/pm_hip.cpp pm/hip/pm_hip.h pm/pm.h
openmp/pm_openmp.o : pm/openmp/pm_openmp.cpp pm/openmp/pm_openmp.h pm/pm.h
host/pm_host.o : pm/host/pm_host.cpp pm/host/pm_host.h pm/pm.h

device/device.o : device/device.cpp device/device.h device/device_context.h device/device_comm.h device/device_cache.h device/device_utilities.h device/device_pdft.h device/device_jk.h device/device_impham.h device/device_lassi.h device/device_h2eff.h device/device_ao2mo.h device/device_fci.h

device/device_jk.o device/device_ao2mo.o device/device_h2eff.o device/device_impham.o device/device_pdft.o device/device_fci.o device/device_lassi.o device/device_comm.o device/device_cache.o device/device_utilities.o : device/device.h device/device_context.h device/device_comm.h device/device_cache.h device/device_utilities.h device/device_pdft.h device/device_jk.h device/device_impham.h device/device_lassi.h device/device_h2eff.h device/device_ao2mo.h device/device_fci.h

cuda/jk.o cuda/ao2mo.o cuda/h2eff.o cuda/impham.o cuda/pdft.o cuda/fci.o cuda/utilities.o : device/device.h device/device_utilities.h device/device_pdft.h device/device_jk.h device/device_impham.h device/device_h2eff.h device/device_ao2mo.h device/device_fci.h cuda/pm_cuda.o
sycl/jk.o sycl/ao2mo.o sycl/h2eff.o sycl/impham.o sycl/pdft.o sycl/fci.o sycl/utilities.o : device/device.h device/device_utilities.h device/device_pdft.h device/device_jk.h device/device_impham.h device/device_h2eff.h device/device_ao2mo.h device/device_fci.h sycl/pm_sycl.o
hip/jk.o hip/ao2mo.o hip/h2eff.o hip/impham.o hip/pdft.o hip/fci.o hip/utilities.o : device/device.h device/device_utilities.h device/device_pdft.h device/device_jk.h device/device_impham.h device/device_h2eff.h device/device_ao2mo.h device/device_fci.h hip/pm_hip.o
openmp/jk.o : device/device.h device/device_pdft.h device/device_jk.h device/device_impham.h device/device_h2eff.h device/device_ao2mo.h device/device_fci.h openmp/pm_openmp.o
host/jk.o host/ao2mo.o host/h2eff.o host/pdft.o host/fci.o host/utilities.o : device/device.h device/device_utilities.h device/device_pdft.h device/device_jk.h device/device_impham.h device/device_h2eff.h device/device_ao2mo.h device/device_fci.h host/pm_host.o

mathlib_cublas.o : mathlib/mathlib_cublas.cpp mathlib/mathlib_cublas.h mathlib/mathlib.h cuda/pm_cuda.o
mathlib_mkl.o : mathlib/mathlib_mkl.cpp mathlib/mathlib_mkl.h mathlib/mathlib.h cuda/pm_cuda.o sycl/pm_sycl.o
mathlib_rocm.o : mathlib/mathlib_rocm.cpp mathlib/mathlib_rocm.h mathlib/mathlib.h hip/pm_hip.o
mathlib_host.o : mathlib/mathlib_host.cpp mathlib/mathlib_host.h mathlib/mathlib.h host/pm_host.o
