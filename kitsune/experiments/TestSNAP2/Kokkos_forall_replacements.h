// ----------------------------------------------------------------------
// Copyright (2019) Sandia Corporation.
// Under the terms of Contract DE-AC04-94AL85000
// with Sandia Corporation, the U.S. Government
// retains certain rights in this software. This
// software is distributed under the Zero Clause
// BSD License
//
// TestSNAP - A prototype for the SNAP force kernel
// Version 0.0.3
// Main changes: GPU AoSoA data layout, optimized recursive polynomial evaluation
//
// Original author: Aidan P. Thompson, athomps@sandia.gov
// http://www.cs.sandia.gov/~athomps, Sandia National Laboratories
//
// Additional authors:
// Sarah Anderson
// Rahul Gayatri
// Steve Plimpton
// Christian Trott
// Evan Weinberg
//
// Collaborators:
// Stan Moore
// Nick Lubbers
// Mitch Wood
//
// ----------------------------------------------------------------------

/* #include <Kokkos_Core.hpp> */
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <string>
#include "kitrt/kitcuda/cuda.h"

template<typename T>
void atomicAdd(T* dest, T val) {
  // TODO: Fix this code to add val into *dest atomically.  Note that
  // this requires atomic-add support for doubles.
  // std::atomic<T>::fetch_add((_Atomic T*)dest, val);
  *dest += val;
}

/* #if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || \ */
/*     defined(KOKKOS_ENABLE_SYCL) || defined(KOKKOS_ENABLE_OPENMPTARGET) */
#define SNAP_ENABLE_GPU
/* #endif */
#define KOKKOS_INLINE_FUNCTION inline

typedef double SNADOUBLE;
struct alignas(2 * sizeof(SNADOUBLE)) SNAcomplex
{
  SNADOUBLE re, im;

  KOKKOS_INLINE_FUNCTION
  SNAcomplex()
    : re(0)
    , im(0)
  {
    ;
  }

  KOKKOS_INLINE_FUNCTION
  SNAcomplex(SNADOUBLE real_in, SNADOUBLE imag_in)
    : re(real_in)
    , im(imag_in)
  {
    ;
  }

  KOKKOS_INLINE_FUNCTION
  void operator=(const SNAcomplex src)
  {
    re = src.re;
    im = src.im;
  }

  KOKKOS_INLINE_FUNCTION
  void operator+=(const SNAcomplex src)
  {
    re += src.re;
    im += src.im;
  }

  KOKKOS_INLINE_FUNCTION
  void operator*=(const SNAcomplex src)
  {
    re += src.re;
    im += src.im;
  }
};

// Struct used to "unfold" ulisttot on the gpu
struct alignas(8) FullHalfMap {
  int idxu_half;
  int flip_sign; // 0 -> isn't flipped, 1 -> flip sign of imaginary, -1 -> flip sign of real
};

/* using ExecSpace = Kokkos::DefaultExecutionSpace; */
/* using HostExecSpace = Kokkos::DefaultHostExecutionSpace; */
/* using Layout = ExecSpace::array_layout; */
/* using MemSpace = ExecSpace::memory_space; */

// using Kokkos::parallel_for;
// using Kokkos::parallel_reduce;
// using Kokkos::View;
// using team_policy = Kokkos::TeamPolicy<ExecSpace>;
// using member_type = team_policy::member_type;
// using Kokkos::PerTeam;
// using Kokkos::TeamThreadRange;
// using Kokkos::ThreadVectorRange;

template<typename T> class View1D {
public:
  T *view = nullptr;
  size_t size = 0;

  View1D() = default;
  View1D([[gnu::unused]] std::string ignored, size_t s) : size(s) {
    if (!s) return;
    view = (T *)__kitrt_cuMemAllocManaged(sizeof(T) * size);
  }
  ~View1D() {
    if (view)
      __kitrt_cuMemFree((void *)view);
    view = nullptr;
  }

  // T &operator()(size_t idx) { return view[idx]; }
  T &operator()(size_t idx) const { return view[idx]; }

  // T &operator[](size_t idx) { return view[idx]; }
  T &operator[](size_t idx) const { return view[idx]; }

  View1D &operator=(const View1D &copy) {
    if (view)
      __kitrt_cuMemFree((void *)view);
    view = copy.view;
    size = copy.size;
    return *this;
  }
  View1D &operator=(View1D &&move) {
    if (view)
      __kitrt_cuMemFree((void *)view);
    view = move.view;
    size = move.size;
    move.view = nullptr;
    return *this;
  }

  T *data(void) const { return view; }

  size_t span(void) const { return size; }
};

template<typename T> class View2D {
public:
  T *view = nullptr;
  size_t size1 = 0, size2 = 0;

  View2D() = default;
  View2D([[gnu::unused]] std::string ignored, size_t s1, size_t s2)
    : size1(s1), size2(s2) {
    if (!(s1 * s2)) return;
    view = (T *)__kitrt_cuMemAllocManaged(sizeof(T) * size1 * size2);
  }
  ~View2D() {
    if (view)
      __kitrt_cuMemFree((void *)view);
    view = nullptr;
  }

  // T &operator()(size_t idx1, size_t idx2) {
  //   return view[idx1 * size2 + idx2];
  // }
  T &operator()(size_t idx1, size_t idx2) const {
    return view[idx1 * size2 + idx2];
  }

  View2D &operator=(const View2D &copy) {
    if (view)
      __kitrt_cuMemFree((void *)view);
    view = copy.view;
    size1 = copy.size1;
    size2 = copy.size2;
    return *this;
  }
  View2D &operator=(View2D &&move) {
    if (view)
      __kitrt_cuMemFree((void *)view);
    view = move.view;
    size1 = move.size1;
    size2 = move.size2;
    move.view = nullptr;
    return *this;
  }

  size_t span(void) const { return size1 * size2; }

  void resize(size_t s1, size_t s2) {
    T *new_view = (T *)__kitrt_cuMemAllocManaged(sizeof(T) * s1 * s2);

    if (view) {
      size_t min_s1 = (size1 > s1) ? s1 : size1;
      size_t min_s2 = (size2 > s2) ? s2 : size2;
      for (size_t i = 0; i < min_s1; ++i)
	for (size_t j = 0; j < min_s2; ++j)
	  new_view[i * s2 + j] = view[i * size2 + j];

      __kitrt_cuMemFree((void *)view);
    }

    view = new_view;
    size1 = s1;
    size2 = s2;
  }
};

template<typename T> class View3D {
public:
  T *view = nullptr;
  size_t size1 = 0, size2 = 0, size3 = 0;

  View3D() = default;
  View3D([[gnu::unused]] std::string ignored, size_t s1, size_t s2, size_t s3)
    : size1(s1), size2(s2), size3(s3) {
    if (!(s1 * s2 * s3)) return;
    view = (T *)__kitrt_cuMemAllocManaged(sizeof(T) * size1 * size2 * size3);
  }
  ~View3D() {
    if (view)
      __kitrt_cuMemFree((void *)view);
    view = nullptr;
  }

  // T &operator()(size_t idx1, size_t idx2, size_t idx3) {
  //   return view[(idx1 * size2 * size3) + (idx2 * size3) + idx3];
  // }
  T &operator()(size_t idx1, size_t idx2, size_t idx3) const {
    return view[(idx1 * size2 * size3) + (idx2 * size3) + idx3];
  }

  View3D &operator=(const View3D &copy) {
    if (view)
      __kitrt_cuMemFree((void *)view);
    view = copy.view;
    size1 = copy.size1;
    size2 = copy.size2;
    size3 = copy.size3;
    return *this;
  }
  View3D &operator=(View3D &&move) {
    if (view)
      __kitrt_cuMemFree((void *)view);
    view = move.view;
    size1 = move.size1;
    size2 = move.size2;
    size3 = move.size3;
    move.view = nullptr;
    return *this;
  }

  size_t span(void) const { return size1 * size2 * size3; }

  void resize(size_t s1, size_t s2, size_t s3) {
    T *new_view = (T *)__kitrt_cuMemAllocManaged(sizeof(T) * s1 * s2 * s3);

    
    if (view) {
      size_t min_s1 = (size1 > s1) ? s1 : size1;
      size_t min_s2 = (size2 > s2) ? s2 : size2;
      size_t min_s3 = (size3 > s3) ? s3 : size3;
      for (size_t i = 0; i < min_s1; ++i)
	for (size_t j = 0; j < min_s2; ++j)
	  for (size_t k = 0; j < min_s3; ++j)
	    new_view[(i * s2 * s3) + (j * s3) + k] =
	      view[(i * size2 * size3) + (j * size3) + k];

      __kitrt_cuMemFree((void *)view);
    }

    view = new_view;
    size1 = s1;
    size2 = s2;
    size3 = s3;
  }
};

template<typename T> class View4D {
public:
  T *view = nullptr;
  size_t size1 = 0, size2 = 0, size3 = 0, size4 = 0;

  View4D() = default;
  View4D([[gnu::unused]] std::string ignored, size_t s1, size_t s2, size_t s3, size_t s4)
    : size1(s1), size2(s2), size3(s3), size4(s4) {
    if (!(s1 * s2 *s3 * s4)) return;
    view = (T *)__kitrt_cuMemAllocManaged(sizeof(T) * size1 * size2 * size3 * size4);
  }
  ~View4D() {
    if (view)
      __kitrt_cuMemFree((void *)view);
    view = nullptr;
  }

  // T &operator()(size_t idx1, size_t idx2, size_t idx3, size_t idx4) {
  //   return view[(idx1 * size2 * size3 * size4) + (idx2 * size3 * size4) +
  // 		(idx3 * size4) + idx4];
  // }
  T &operator()(size_t idx1, size_t idx2, size_t idx3, size_t idx4) const {
    return view[(idx1 * size2 * size3 * size4) + (idx2 * size3 * size4) +
		(idx3 * size4) + idx4];
  }

  View4D &operator=(const View4D &copy) {
    if (view)
      __kitrt_cuMemFree((void *)view);
    view = copy.view;
    size1 = copy.size1;
    size2 = copy.size2;
    size3 = copy.size3;
    size4 = copy.size4;
    return *this;
  }
  View4D &operator=(View4D &&move) {
    if (view)
      __kitrt_cuMemFree((void *)view);
    view = move.view;
    size1 = move.size1;
    size2 = move.size2;
    size3 = move.size3;
    size4 = move.size4;
    move.view = nullptr;
    return *this;
  }

  size_t span(void) const { return size1 * size2 * size3 * size4; }
};

// 1D,2D,3D views for int's
/* using int_View1D = View<int*, Layout, MemSpace>; */
/* using int_View2D = View<int**, Layout, MemSpace>; */
/* using int_View3D = View<int***, Layout, MemSpace>; */
/* using int_View2DR = View<int**, Kokkos::LayoutRight, MemSpace>; */
using int_View1D = View1D<int>;
using int_View2D = View2D<int>;
using int_View3D = View3D<int>;
using int_View2DR = View2D<int>;

// Host-View mirrors for int's
/* using HostInt_View1D = int_View1D::HostMirror; */
/* using HostInt_View2D = int_View2D::HostMirror; */
/* using HostInt_View3D = int_View3D::HostMirror; */
/* using HostInt_View2DR = int_View2DR::HostMirror; */
using HostInt_View1D = int_View1D;
using HostInt_View2D = int_View2D;
using HostInt_View3D = int_View3D;
using HostInt_View2DR = int_View2DR;

// 1D,2D,3D views for double's
/* using double_View1D = View<double*, Layout, MemSpace>; */
/* using double_View2D = View<double**, Layout, MemSpace>; */
/* using double_View3D = View<double***, Layout, MemSpace>; */
/* using double_View1DL = View<double*, Kokkos::LayoutLeft, MemSpace>; */
/* using double_View2DL = View<double**, Kokkos::LayoutLeft, MemSpace>; */
/* using double_View3DL = View<double***, Kokkos::LayoutLeft, MemSpace>; */
/* using double_View4DL = View<double****, Kokkos::LayoutLeft, MemSpace>; */
using double_View1D = View1D<double>;
using double_View2D = View2D<double>;
using double_View3D = View3D<double>;
using double_View1DL = View1D<double>;
using double_View2DL = View2D<double>;
using double_View3DL = View3D<double>;
using double_View4DL = View4D<double>;

// Host-View mirrors for double's
/* using HostDouble_View1D = double_View1D::HostMirror; */
/* using HostDouble_View2D = double_View2D::HostMirror; */
/* using HostDouble_View3D = double_View3D::HostMirror; */
using HostDouble_View1D = double_View1D;
using HostDouble_View2D = double_View2D;
using HostDouble_View3D = double_View3D;

// 1D,2D,3D views for SNAcomplex's
/* using SNAcomplex_View2D = View<SNAcomplex**, Layout, MemSpace>; */
/* using SNAcomplex_View3D = View<SNAcomplex***, Layout, MemSpace>; */
/* using SNAcomplex_View4D = View<SNAcomplex****, Layout, MemSpace>; */
/* using SNAcomplex_View2DR = View<SNAcomplex**, Kokkos::LayoutRight, MemSpace>; */
/* using SNAcomplex_View3DR = View<SNAcomplex***, Kokkos::LayoutRight, MemSpace>; */
/* using SNAcomplex_View4DR = View<SNAcomplex****, Kokkos::LayoutRight, MemSpace>; */
/* using SNAcomplex_View2DL = View<SNAcomplex**, Kokkos::LayoutLeft, MemSpace>; */
/* using SNAcomplex_View3DL = View<SNAcomplex***, Kokkos::LayoutLeft, MemSpace>; */
/* using SNAcomplex_View4DL = View<SNAcomplex****, Kokkos::LayoutLeft, MemSpace>; */
using SNAcomplex_View2D = View2D<SNAcomplex>;
using SNAcomplex_View3D = View3D<SNAcomplex>;
using SNAcomplex_View4D = View4D<SNAcomplex>;
using SNAcomplex_View2DR = View2D<SNAcomplex>;
using SNAcomplex_View3DR = View3D<SNAcomplex>;
using SNAcomplex_View4DR = View4D<SNAcomplex>;
using SNAcomplex_View2DL = View2D<SNAcomplex>;
using SNAcomplex_View3DL = View3D<SNAcomplex>;
using SNAcomplex_View4DL = View4D<SNAcomplex>;

// 1D view for mapping from compressed ulisttot -> full
/* using FullHalfMap_View1D = View<FullHalfMap*, Layout, MemSpace>; */
using FullHalfMap_View1D = View1D<FullHalfMap>;

// Host-View mirrors
/* using HostFullHalfMap_View1D = FullHalfMap_View1D::HostMirror; */
using HostFullHalfMap_View1D = FullHalfMap_View1D;

// scratch memory views for SNAcomplex
/* using ScratchViewType = View<SNAcomplex *, ExecSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>; */
// TODO: We should use device-side scratch memory for this storage,
// but I'm not sure how to do that right now.
class ScratchViewType {
  using T = SNAcomplex;
public:
  T *view = nullptr;
  size_t size = 0;

  ScratchViewType() = default;
  ScratchViewType(T *preallocated, size_t s) : size(s) {
    if (!s) return;
    view = preallocated;
  }
  ~ScratchViewType() {
    view = nullptr;
  }

  T &operator()(size_t idx) const { return view[idx]; }

  T &operator[](size_t idx) const { return view[idx]; }

  T *data(void) const { return view; }

  size_t span(void) const { return size; }  
};


// Custom replacement for member_type class.
class member_type {
  int _league_rank = 0, _team_size = 0, _team_rank = 0, _per_team_scratch_size = 0;
  SNAcomplex *scratch_alloc = nullptr;
public:
  member_type(int lrank, int tsize, int trank)
    : _league_rank(lrank), _team_size(tsize), _team_rank(trank) {}
  member_type(int lrank, int tsize, int trank, SNAcomplex *scratch,
	      int per_team_scratch_size)
    : _league_rank(lrank), _team_size(tsize), _team_rank(trank),
      scratch_alloc(scratch), _per_team_scratch_size(per_team_scratch_size) {}

  int league_rank() const { return _league_rank; }
  int team_size() const { return _team_size; }
  int team_rank() const { return _team_rank; }

  SNAcomplex *team_scratch(int i) const {
    return &scratch_alloc[_league_rank * _per_team_scratch_size];
  }
};

// template<typename T> T *create_mirror_view(T *view) { return view; }
#define create_mirror_view(v) v
