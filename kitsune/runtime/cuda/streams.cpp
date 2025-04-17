//===- streams.cpp - Kitsune runtime CUDA streams support    --------------===//
// Copyright (c) 2021, 2023 Los Alamos National Security, LLC.
//
// All rights reserved.
//
//  Copyright 2021. Los Alamos National Security, LLC. This software was
//  produced under U.S. Government contract DE-AC52-06NA25396 for Los
//  Alamos National Laboratory (LANL), which is operated by Los Alamos
//  National Security, LLC for the U.S. Department of Energy. The
//  U.S. Government has rights to use, reproduce, and distribute this
//  software.  NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY,
//  LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY
//  FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
//  derivative works, such modified software should be clearly marked,
//  so as not to confuse it with the version available from LANL.
//
//  Additionally, redistribution and use in source and binary forms,
//  with or without modification, are permitted provided that the
//  following conditions are met:
//
//    * Redistributions of source code must retain the above copyright
//      notice, this list of conditions and the following disclaimer.
//
//    * Redistributions in binary form must reproduce the above
//      copyright notice, this list of conditions and the following
//      disclaimer in the documentation and/or other materials provided
//      with the distribution.
//
//    * Neither the name of Los Alamos National Security, LLC, Los
//      Alamos National Laboratory, LANL, the U.S. Government, nor the
//      names of its contributors may be used to endorse or promote
//      products derived from this software without specific prior
//      written permission.
//
//  THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND
//  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
//  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
//  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//  DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR
//  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
//  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
//  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
//  USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
//  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
//  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
//  OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
//  SUCH DAMAGE.
//
//===----------------------------------------------------------------------===//

#include "kitcuda.h"
#include "kitcuda_dylib.h"
#include <algorithm>
#include <deque>
#include <mutex>
#include <optional>
#include <stdio.h>
#include <sys/syscall.h>
#include <unistd.h>

extern "C" { // Taken from cheetah
typedef void (*__cilk_identity_fn)(void *);
typedef void (*__cilk_reduce_fn)(void *, void *);

void *__cilkrts_reducer_lookup(void *key, size_t size, __cilk_identity_fn id,
                               __cilk_reduce_fn reduce);
void __cilkrts_reducer_register(void *key, size_t size, __cilk_identity_fn id,
                                __cilk_reduce_fn reduce);
void __cilkrts_reducer_unregister(void *key);
}

// On older systems gettid() is not exposed and the syscall()
// interface must be used.  It won't hurt us to just use that
// everywhere as there are still systems in use with older
// system libraries that are missing the call...
#define gettid() syscall(SYS_gettid)

// Stream creation can be expensive.  We "recycle" them when possible.
struct KitCudaStreamEvent {
  CUstream stream;
  CUevent event;
};

typedef kitrt::deque<KitCudaStreamEvent *> KitCudaStreamList;
static KitCudaStreamList _kitcuda_streams;
static std::mutex _kitcuda_stream_mutex;
static KitCudaStreamEvent *_kitcuda_stream_event;

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#endif

#define KITCUDA_ASYNC_STREAM_JOIN

KitCudaStreamEvent *__kitcuda_stream_event_claim() {
  KitCudaStreamEvent *stream_event;
  _kitcuda_stream_mutex.lock();
  if (!_kitcuda_streams.empty()) {
    stream_event = _kitcuda_streams.front();
    _kitcuda_streams.pop_front();
    _kitcuda_stream_mutex.unlock();
    if (__kitrt_verbose_mode())
      fprintf(stderr, "reusing thread stream.\n");
  } else {
    _kitcuda_stream_mutex.unlock();
    stream_event = reinterpret_cast<KitCudaStreamEvent *>(
        malloc(sizeof(KitCudaStreamEvent)));
    if (__kitrt_verbose_mode())
      fprintf(stderr, "creating new thread stream.\n");
    CU_SAFE_CALL(
        cuStreamCreate_p(&stream_event->stream, CU_STREAM_NON_BLOCKING));
    CU_SAFE_CALL(cuEventCreate_p(&stream_event->event, CU_EVENT_DEFAULT));
  }
  return stream_event;
}

void __kitcuda_stream_event_identity(void *se_raw) {
  auto *se = reinterpret_cast<KitCudaStreamEvent **>(se_raw);
  *se = __kitcuda_stream_event_claim();
}

void __kitcuda_stream_event_recycle(KitCudaStreamEvent *se) {
  _kitcuda_stream_mutex.lock();
  _kitcuda_streams.push_back(se);
  _kitcuda_stream_mutex.unlock();
}

void __kitcuda_stream_event_merge(void *se_left_raw, void *se_right_raw) {
  auto **se_left = reinterpret_cast<KitCudaStreamEvent **>(se_left_raw);
  auto **se_right = reinterpret_cast<KitCudaStreamEvent **>(se_right_raw);

#ifdef KITCUDA_ASYNC_STREAM_JOIN
  CU_SAFE_CALL(cuEventRecord_p((*se_right)->event, (*se_right)->stream));
  CU_SAFE_CALL(cuStreamWaitEvent_p((*se_left)->stream, (*se_right)->event,
                                   CU_EVENT_WAIT_DEFAULT));
  // CU_SAFE_CALL(cuLaunchHostFunc_p(
  //     (*se_left)->stream,
  //     reinterpret_cast<CUhostFn>(__kitcuda_stream_event_recycle),
  //     (void *)*se_right));
#else
  CU_SAFE_CALL(cuStreamSynchronize_p((*se_right)->stream));
#endif
  __kitcuda_stream_event_recycle(*se_right);

  *se_right = nullptr;
}

void *__kitcuda_get_thread_stream() {
  KIT_NVTX_PUSH("kitcuda:get_thread_stream", KIT_NVTX_STREAM);
  CUstream cu_stream;
  auto **se = reinterpret_cast<KitCudaStreamEvent **>(__cilkrts_reducer_lookup(
      &_kitcuda_stream_event, sizeof(_kitcuda_stream_event),
      __kitcuda_stream_event_identity, __kitcuda_stream_event_merge));
  cu_stream = (*se)->stream;
  KIT_NVTX_POP();
  if (__kitrt_verbose_mode())
    fprintf(stderr, "returning thread stream: %p\n", cu_stream);
  return (void *)cu_stream;
}

void __kitcuda_sync_thread_stream(void *opaque_stream) {
  assert(opaque_stream != nullptr && "unexpected null stream pointer!");
  KIT_NVTX_PUSH("kitcuda:sync_thread_stream", KIT_NVTX_STREAM);
  CUstream stream = (CUstream)opaque_stream;
  CU_SAFE_CALL(cuStreamSynchronize_p(stream));
  KIT_NVTX_POP();
}

void __kitcuda_local_stream_begin() {}

void __kitcuda_local_stream_end() {}

void __kitcuda_sync_current_stream() {
  KIT_NVTX_PUSH("kitcuda:sync_current_stream", KIT_NVTX_STREAM);
  CUstream stream = (CUstream)__kitcuda_get_thread_stream();
  CU_SAFE_CALL(cuStreamSynchronize_p(stream));
  KIT_NVTX_POP();
}

void __kitcuda_sync_context() {
  KIT_NVTX_PUSH("kitcuda:sync_context", KIT_NVTX_STREAM);
  CUcontext ctx;
  __kitcuda_set_context();
  CU_SAFE_CALL(cuCtxSynchronize_p());
  KIT_NVTX_POP();
}

void __kitcuda_initialize_thread_streams() {
  _kitcuda_stream_event = __kitcuda_stream_event_claim();
  __cilkrts_reducer_register(
      &_kitcuda_stream_event, sizeof(_kitcuda_stream_event),
      __kitcuda_stream_event_identity, __kitcuda_stream_event_merge);
}

void __kitcuda_destroy_thread_streams() {
  KIT_NVTX_PUSH("kitrt:delete_thread_streams", KIT_NVTX_STREAM);
  _kitcuda_stream_mutex.lock();
  for (auto &entry : _kitcuda_streams) {
    CU_SAFE_CALL(cuStreamDestroy_v2_p(entry->stream));
    CU_SAFE_CALL(cuEventDestroy_v2_p(entry->event));
    free(entry);
  }
  _kitcuda_streams.clear();
  _kitcuda_stream_mutex.unlock();
  __cilkrts_reducer_unregister(&_kitcuda_stream_event);
  CU_SAFE_CALL(cuStreamDestroy_v2_p(_kitcuda_stream_event->stream));
  CU_SAFE_CALL(cuEventDestroy_v2_p(_kitcuda_stream_event->event));
  free(_kitcuda_stream_event);
  _kitcuda_stream_event = nullptr;
  KIT_NVTX_POP();
}

} // extern "C"
