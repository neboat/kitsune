//===- IntrinsicsKitsune.td - Defines Kitsune intrinsics ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Kitsune-specific intrinsics. In the intrinsics here,
// boolean arguments have type i8 instead of i1 to simplify the lowering since,
// in the runtime, booleans are always 8-bit.
//
//===----------------------------------------------------------------------===//
//
// NOTES
//
// - Intrinsics that are asynchronous should start with kit_async. Other parts
//   of the code expect this. Failing to do so could result in miscompilation
//   since analyses may make incorrect assumptions about data being available
//   when reordering kernel launches.
//
//===----------------------------------------------------------------------===//

// Pointers to data in Kitsune's mobile address space.
def llvm_mobile_ptr_ty : LLVMQualPointerType<67>;

//===----------------------------------------------------------------------===//

// Launch a function on one or more CPU threads. An opaque context object is
// returned which must be used to join the threads.
def int_kit_async_launch_threads
    : Intrinsic<[llvm_ptr_ty], // thread context
                [llvm_i32_ty,  // tapir target id
                 llvm_ptr_ty,  // thread function to launch on each thread
                 llvm_i64_ty,  // start index passed to the thread function
                 llvm_i64_ty,  // end index passed to the thread function
                 llvm_i64_ty,  // grain size passed to the thread function
                 llvm_ptr_ty], // argument bundle passed to the thread function
                [NonNull<RetIndex>, Dereferenceable<RetIndex, 16>,
                 ImmArg<ArgIndex<0>>,
                 NonNull<ArgIndex<1>>,
                 NonNull<ArgIndex<5>>,
                 IntrInaccessibleMemOrArgMemOnly, IntrHasSideEffects]>;

// Sync the threads launched in a previous call to kit.async.launch.threads
def int_kit_sync_threads
    : Intrinsic<[],
                [llvm_i32_ty,  // tapir target id
                 llvm_ptr_ty], // thread context
                [ImmArg<ArgIndex<0>>,
                 NonNull<ArgIndex<1>>, Dereferenceable<ArgIndex<1>, 16>,
                 IntrInaccessibleMemOrArgMemOnly, IntrHasSideEffects]>;

// Launch a kernel on the GPU on the given stream. Returns the same stream. If
// a stream is not provided, one will be created and returned. The stream is a
// pointer to an opaque object.
def int_kit_async_launch_kernel
    : Intrinsic<[llvm_ptr_ty], // stream
                [llvm_i32_ty,  // tapir target id
                 llvm_ptr_ty,  // fat binary
                 llvm_ptr_ty,  // name
                 llvm_i64_ty,  // trip_count
                 llvm_i32_ty,  // threads per block
                 llvm_ptr_ty,  // instruction mix info
                 llvm_ptr_ty,  // stream
                 llvm_vararg_ty],
                [ImmArg<ArgIndex<0>>, IntrInaccessibleMemOrArgMemOnly,
                 IntrHasSideEffects]>;

// Copy bytes from the GPU to the host. This is a non-blocking call i.e. the
// memory will be copied asyncrhonously.
def int_kit_async_memcpy_dtoh
    : Intrinsic<[],
                [llvm_i32_ty,  // tapir target id
                 llvm_ptr_ty,  // pointer to host memory (destination)
                 llvm_ptr_ty,  // pointer to device memory (source)
                 llvm_i64_ty,  // number of bytes to copy
                 llvm_ptr_ty], // the stream on which to perform the copy
                [ImmArg<ArgIndex<0>>, IntrInaccessibleMemOrArgMemOnly,
                 IntrHasSideEffects]>;

// Copy bytes from the host to the GPU. This is a non-blocking call i.e. the
// memory will be copied asynchronously.
def int_kit_async_memcpy_htod
    : Intrinsic<[],
                [llvm_i32_ty,  // tapir target id
                 llvm_ptr_ty,  // pointer to device memory (destination)
                 llvm_ptr_ty,  // pointer to host memory (source)
                 llvm_i64_ty,  // number of bytes to copy
                 llvm_ptr_ty], // the stream on which to perform the copy
                [ImmArg<ArgIndex<0>>, IntrInaccessibleMemOrArgMemOnly,
                 IntrHasSideEffects]>;

// Issue a prefetch from the GPU to the host. This will be lowered to an
// appropriate call in Kitusne's runtime. The function takes a pointer to
// an opaque stream on which to issue the prefetch, or null to indicate that a
// new stream should be created. Returns the pointer to the stream - if one was
// given - or a new stream that was created. A pointer to the buffer to prefetch
// is also provided. The number of bytes to prefetch may be -1 which indicates
// that the runtime is required to determine the number of bytes to prefetch.
def int_kit_async_prefetch_dtoh
    : Intrinsic<[llvm_ptr_ty], // stream
                [llvm_i32_ty,  // tapir target id
                 llvm_ptr_ty,  // pointer to memory to prefetch
                 llvm_i64_ty,  // number of bytes to prefetch
                 llvm_ptr_ty], // stream
                [ImmArg<ArgIndex<0>>, IntrInaccessibleMemOrArgMemOnly,
                 IntrHasSideEffects]>;

// Issue a prefetch from the host to the GPU. This will be lowered to an
// appropriate call in Kitsune's runtime. The function takes a pointer to
// an opaque stream on which to issue the prefetch, or null to indicate that a
// new stream should be created. Returns the pointer to the stream - if one was
// given - or a new stream that was created. A pointer to the buffer to prefetch
// is also provided. The number of bytes to prefetch may be -1 which indicates
// that the runtime is required to determine the number of bytes to prefetch.
def int_kit_async_prefetch_htod
    : Intrinsic<[llvm_ptr_ty], // stream
                [llvm_i32_ty,  // tapir target id
                 llvm_ptr_ty,  // pointer to memory to prefetch
                 llvm_i64_ty,  // number of bytes to prefetch
                 llvm_ptr_ty], // stream
                [ImmArg<ArgIndex<0>>, IntrInaccessibleMemOrArgMemOnly,
                 IntrHasSideEffects, IntrWillReturn]>;

// Mobile memory allocator. This will allocate the given number of bytes as a
// mobile buffer. The lowering of this intrinsic will depend on
//
//   - The Tapir target(s) that are enabled
//   - The uses of the buffer
//
// TODO: This should eventually return a pointer in a specific address space.
// This may require the returned type to be replaced with an
// LLVMQualPointerType.
def int_kit_mobile_alloc
    : Intrinsic<[llvm_mobile_ptr_ty], [llvm_i64_ty],
                [IntrInaccessibleMemOnly, NoAlias<RetIndex>]>;

// Mobile memory deallocator. This will free memory that was previously
// allocated by kitsune_mobile_alloc.
def int_kit_mobile_free
    : Intrinsic<[], [llvm_mobile_ptr_ty],
                [IntrInaccessibleMemOrArgMemOnly,
                 NoCapture<ArgIndex<0>>, ReadNone<ArgIndex<0>>]>;

def int_kit_mobile_realloc
    : Intrinsic<[llvm_mobile_ptr_ty], [llvm_mobile_ptr_ty, llvm_i64_ty],
                [IntrInaccessibleMemOrArgMemOnly, NoCapture<ArgIndex<0>>,
                 ReadNone<ArgIndex<0>>, NoAlias<RetIndex>]>;

// Enable launch refinement. Setting this will cause the runtime to calculate
// a kernel's launch parameters based on some heuristic. In the case of NVIDIA
// GPU's, this is an occupancy-based heuristic provided by cuda's API.
//
// NOTE: As of May 2025, this is only implemented in kitsune's cuda runtime. If
// this is implemented for kitsune's hip runtime as well, this comment should be
// updated.
def int_kit_enable_refine_launches
    : Intrinsic<[],
                [llvm_i32_ty, // tapir target id
                 llvm_i8_ty], // Non-zero to refine launches, zero otherwise
                [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>,
                 IntrInaccessibleMemOnly, IntrHasSideEffects]>;

// Call a function in kitsune's runtime that enables verbose mode.
def int_kit_enable_verbose
    : Intrinsic<[],
                [llvm_i8_ty], // Non-zero to enable verbose mode, zero otherwise
                [ImmArg<ArgIndex<0>>, IntrInaccessibleMemOrArgMemOnly,
                 IntrHasSideEffects]>;

// Enable xnack in the runtime. This is only relevant for the hip tapir target
// and its associated kitsune runtime.
def int_kit_enable_xnack
    : Intrinsic<[],
                [llvm_i8_ty], // Non-zero to enable xnack, zero otherwise
                [IntrInaccessibleMemOnly, IntrHasSideEffects]>;

// Enable the y-axis launch pattern for all GPU kernels.
//
// NOTE: As of May 2025, this is only implemented in kitsune's hip runtime. If
// this is implemented for kitsune's cuda runtime as well, this comment should
// be updated.
def int_kit_enable_y_axis_launches
    : Intrinsic<[],
                [llvm_i32_ty, // tapir target id
                 llvm_i8_ty], // Non-zero to enable y-axis launches, 0 otherwise
                [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>,
                 IntrInaccessibleMemOnly, IntrHasSideEffects]>;

// Call the appropriate finalization function from the kitsune runtime.
def int_kit_finalize
    : Intrinsic<[],
                [llvm_i32_ty], // tapir target id
                [ImmArg<ArgIndex<0>>, IntrInaccessibleMemOnly,
                 IntrHasSideEffects]>;

// Initialize the appropriate kitsune runtime. There are separate initialization
// functions in kitrt for the runtime components of certain tapir targets.
def int_kit_initialize
    : Intrinsic<[],
                [llvm_i32_ty], // tapir target id
                [ImmArg<ArgIndex<0>>, IntrInaccessibleMemOnly,
                 IntrHasSideEffects]>;

// Copy bytes from the GPU to the host. This is a blocking call.
def int_kit_memcpy_dtoh
    : Intrinsic<[],
                [llvm_i32_ty,  // tapir target id
                 llvm_ptr_ty,  // pointer to host memory (destination)
                 llvm_ptr_ty,  // pointer to device memory (source)
                 llvm_i64_ty], // number of bytes to copy
                [ImmArg<ArgIndex<0>>, IntrInaccessibleMemOrArgMemOnly,
                 IntrHasSideEffects]>;

// Copy bytes from the host to the GPU. This is a blocking call.
def int_kit_memcpy_htod
    : Intrinsic<[],
                [llvm_i32_ty,  // tapir target id
                 llvm_ptr_ty,  // pointer to device memory (destination)
                 llvm_ptr_ty,  // pointer to host memory (source)
                 llvm_i64_ty], // number of bytes to copy
                [ImmArg<ArgIndex<0>>, IntrInaccessibleMemOrArgMemOnly,
                 IntrHasSideEffects]>;

// Set a fixed number of threads per block to use for all kernels launched by
// kitsune's runtime.
def int_kit_set_fixed_tpb
    : Intrinsic<[],
                [llvm_i32_ty,  // tapir target id
                 llvm_i32_ty], // threads per block
                [ImmArg<ArgIndex<0>>, IntrInaccessibleMemOnly,
                 IntrHasSideEffects]>;

// Set the maximum number of threads per block to use for all kernels launched
// by kitsune's runtime.
def int_kit_set_max_tpb
    : Intrinsic<[],
                [llvm_i32_ty,  // tapir target id
                 llvm_i32_ty], // threads per block
                [ImmArg<ArgIndex<0>>, IntrInaccessibleMemOnly,
                 IntrHasSideEffects]>;

// Get a pointer to a symbol in the device. This is typically used to get the
// address of a global variable, but in principle, it could be used with any
// global value.
def int_kit_symbol_device_ptr
    : Intrinsic<[llvm_ptr_ty], // device pointer
                [llvm_i32_ty,  // tapir target id
                 llvm_ptr_ty,  // fat binary
                 llvm_ptr_ty], // name
                [ImmArg<ArgIndex<0>>, IntrInaccessibleMemOrArgMemOnly,
                 IntrHasSideEffects]>;

// Copy a symbol from device to host.
// FIXME: The runtime call here simply calls cuMemcpyDtoH. We should consider
// removing this and using kit_memcpy_dtoh instead.
def int_kit_symbol_memcpy_dtoh
    : Intrinsic<[],
                [llvm_i32_ty,  // tapir target id
                 llvm_ptr_ty,  // pointer to symbol on the host
                 llvm_ptr_ty,  // pointer to symbol on the device
                 llvm_i64_ty], // number of bytes to copy
                [ImmArg<ArgIndex<0>>, IntrInaccessibleMemOrArgMemOnly,
                 IntrHasSideEffects]>;

// Copy a symbol from host to device.
// FIXME: The runtime call here simply calls cuMemcpyHtoD. We should consider
// removing this and using kit_memcpy_htod instead.
def int_kit_symbol_memcpy_htod
    : Intrinsic<[],
                [llvm_i32_ty,  // tapir target id
                 llvm_ptr_ty,  // pointer to symbol on the device
                 llvm_ptr_ty,  // pointer to symbol on the host
                 llvm_i64_ty], // number of bytes to copy
                [ImmArg<ArgIndex<0>>, IntrInaccessibleMemOrArgMemOnly,
                 IntrHasSideEffects]>;

// Synchronize a thread stream.
def int_kit_sync_stream
    : Intrinsic<[],
                [llvm_i32_ty,  // tapir target id
                 llvm_ptr_ty], // stream
                [ImmArg<ArgIndex<0>>, IntrInaccessibleMemOrArgMemOnly,
                 IntrHasSideEffects]>;

// Get a thread stream for use.
def int_kit_thread_stream
    : Intrinsic<[llvm_ptr_ty],
                [llvm_i32_ty], // tapir target id
                [ImmArg<ArgIndex<0>>, IntrInaccessibleMemOrArgMemOnly,
                 IntrHasSideEffects]>;
