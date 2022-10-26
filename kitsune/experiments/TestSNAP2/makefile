include ../common.mk
include ../cuda.mk
include ../kokkos.mk
include ../gcc.mk

# General (shared) flags (see common.mk)
clang_flags=${cxx_flags} ${opt_flags} ${clang_info_flags}
# kitsune kokkos mode flags.

#### Kitsune+Tapir options
#  Notes:
#   -ftapir=cuda will target the CUDA ABI transform.
#   -cuabi-verbose dumps kernel stats via ptxas output.
#   -cuabi-keep-files saves intermediate files (.ll, .ptx, .fatbin) on disk for a closer look at codegen details.
#
kitsune_kokkos_flags = -fkokkos -fkokkos-no-init -I${kitsune_prefix}/include
tapir_cu_launch_flags=-I${CUDA_HOME}/include
tapir_cu_flags=-ftapir=cuda -mllvm -cuabi-verbose -mllvm -cuabi-arch=${NVARCH} ${tapir_cu_launch_flags} -mllvm -cuabi-opt-level=3 -mllvm -cuabi-keep-files
# tapir_cu_flags=-ftapir=cuda -ffp-contract=fast -mllvm -cuabi-arch=${NVARCH} ${tapir_cu_launch_flags} -mllvm -cuabi-opt-level=3 -mllvm -cuabi-run-post-opts=false -mllvm -cuabi-disable-prefetch=true
tapir_stripmine_flags=-mllvm -stripmine-count=1 -mllvm -stripmine-coarsen-factor=1 -mllvm -cuabi-default-grainsize=1
####

#### CUDA flags
nvcc_flags=--std c++17 -m64 -arch=${NVARCH} --resource-usage ${opt_flags} --no-exceptions -I${kitsune_prefix}/include
nvcc_kokkos_flags=${nvcc_flags} -I${kokkos_cuda_prefix}/include -I${kitsune_prefix}/include
clang_cuda_flags=-xcuda --cuda-gpu-arch=${NVARCH} ${cxx_flags} ${opt_flags} -I${kitsune_prefix}/include
####

#### Kokkos stuff
# kokkos_cu_flags=-I${kokkos_cu_prefix}/include
kokkos_cu_flags=${kokkos_cxx_flags}
kokkos_clang_flags=${clang_cuda_flags} -I${kokkos_cu_prefix}/include
kokkos_nvcc_flags=-I${kokkos_cu_prefix}/include ${nvcc_cxx_flags} ${opt_flags} -m64 -arch=${NVARCH} --resource-usage --no-exceptions -I${kitsune_prefix}/include

#### TestSNAP-specific stuff
ref_data=14
# TODO: Add support to compile the forall code with -ffast-math.
test_snap_flags=-DREFDATA_TWOJ=${ref_data} -std=c++17 -DKOKKOS_ENABLE_CXX17 # -ffast-math

all: test_snap-kokkos.clang.${host_arch} test_snap-forall.cuda.${host_arch}

test_snap-kokkos.clang.${host_arch}: memory-kokkos.clang.${host_arch}.o sna-kokkos.clang.${host_arch}.o test_snap-kokkos.clang.${host_arch}.o
	${clangxx} -o $@ $? ${kokkos_ld_flags} ${kokkos_ld_libs} -Xlinker -rpath=${kokkos_cu_prefix}/lib

test_snap-forall.cuda.${host_arch}: memory.cpp sna-forall.cpp test_snap-forall.cpp 
	${clangxx} ${test_snap_flags} ${clang_flags} ${tapir_cu_flags} ${tapir_stripmine_flags} -o $@ $? -Xlinker -rpath=${kitsune_prefix}/lib

%-kokkos.clang.${host_arch}.o:%.cpp
	${clangxx} ${test_snap_flags} ${clang_cuda_flags} ${kokkos_cu_flags} -c -o $@ $<

# sna-forall.cuda.${host_arch}.o:sna-forall.cpp
# 	${clangxx} ${test_snap_flags} ${clang_flags} ${tapir_cu_flags} ${tapir_stripmine_flags} -c -o $@ $<
# test_snap-forall.cuda.${host_arch}.o:test_snap-forall.cpp
# 	${clangxx} ${test_snap_flags} ${clang_flags} ${tapir_cu_flags} ${tapir_stripmine_flags} -c -o $@ $<
%-forall.cuda.${host_arch}.o:%.cpp
	${clangxx} ${test_snap_flags} ${clang_flags} ${tapir_cu_flags} ${tapir_stripmine_flags} -c -o $@ $<

clean:
	-rm -f *.${host_arch}
	-rm -f *.fatbin
	-rm -rf *-cfg-tmp
	-rm -f *.bc
	-rm -f *.fatbin
	-rm -f *.ppm *.jpg
	-rm -f *.ll *.ptx *.csv *.log *.s *.fbin *.tapir
	-rm -f *.dat
	-rm -f *.o
