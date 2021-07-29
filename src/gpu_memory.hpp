/*
 *  Copyright 2012 Ben Barsdell
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *      distributed under the License is distributed on an "AS IS" BASIS,
 *      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*
  This file just contains crappy wrappers for CUDA memory functions
*/

#pragma once

#include <CL/opencl.hpp>
using namespace cl;
#include "utils.dp.hpp"
// typedef unsigned int gpu_size_t;

template <typename T>
bool malloc_device(cl::Buffer &addr, gpu_size_t count) {
    cl_int error;
    addr = cl::Buffer(dedisp::device_manager::instance().current_context(), CL_MEM_READ_WRITE, count * sizeof(T), nullptr, &error);
	if( error != CL_SUCCESS ) {
		return false;
	}
        return true;
}
template<typename T>
void free_device(cl::Buffer &addr) {
        std::move(addr);
}
template <typename T>
bool copy_host_to_device(cl::Buffer& dst, const T* src, gpu_size_t count,
                         const cl::CommandQueue& stream = dedisp::device_manager::instance().current_queue()) {
    // TODO: Can't use Async versions unless host memory is pinned!
    // TODO: Passing a device pointer as src causes this to segfault
    stream.enqueueWriteBuffer(dst, CL_TRUE, 0, count * sizeof(T), src);
    //#ifdef DEDISP_DEBUG
    cl_int error = stream.finish();
    if (error != CL_SUCCESS) {
        return false;
    }
    //#endif
	return true;
}
template <typename T>
bool copy_device_to_host(T *dst, cl::Buffer& src, gpu_size_t count,
                         const cl::CommandQueue& stream = dedisp::device_manager::instance().current_queue()) {
    // TODO: Can't use Async versions unless host memory is pinned!
    stream.enqueueReadBuffer(src, CL_TRUE, 0, count * sizeof(T), dst);
    //#ifdef DEDISP_DEBUG
    cl_int error = stream.finish();
    if (error != CL_SUCCESS) {
        return false;
    }
    //#endif
	return true;
}
#if 0
// ------- REMOVED --------
template<typename T>
bool copy_host_to_symbol(const char* symbol, const T* src,
						 gpu_size_t count, cudaStream_t stream=0) {
	// TODO: Can't use Async versions unless host memory is pinned!
	cudaMemcpyToSymbol/*Async*/(symbol, src,
							count * sizeof(T),
							0, cudaMemcpyHostToDevice/*,
													   stream*/);
	//#ifdef DEDISP_DEBUG
	cudaThreadSynchronize();
	cudaError_t error = cudaGetLastError();
	if( error != cudaSuccess ) {
		return false;
	}
	//#endif
	return true;
}
template<typename U, typename T>
bool copy_device_to_symbol(/*const char**/U symbol, const T* src,
						   gpu_size_t count, cudaStream_t stream=0) {
	cudaMemcpyToSymbolAsync(symbol, src,
							count * sizeof(T),
							0, cudaMemcpyDeviceToDevice,
							stream);
	//#ifdef DEDISP_DEBUG
	cudaThreadSynchronize();
	cudaError_t error = cudaGetLastError();
	if( error != cudaSuccess ) {
		return false;
	}
	//#endif
	return true;
}
// ------- REMOVED --------
#endif
// Note: Strides must be given in units of bytes
template </*typename T, */typename U>
bool copy_host_to_device_2d(cl::Buffer& dst, gpu_size_t dst_offset, gpu_size_t dst_stride,
                            const U *src, gpu_size_t src_offset, gpu_size_t src_stride,
                            gpu_size_t width_bytes, gpu_size_t height,
                            const cl::CommandQueue& stream = dedisp::device_manager::instance().current_queue()) {
    // TODO: Can't use Async versions unless host memory is pinned!
    cl_int error = 
    stream.enqueueWriteBufferRect(dst, /* blocking = */ CL_TRUE,
                                  /* buffer_offset = */ {dst_offset, 0, 0},
                                  /* host_offset = */ {src_stride, 0, 0},
                                  /* region = */ {width_bytes, height, 1},
                                  /* buffer_row_pitch = */ dst_stride,
                                  /* buffer_slice_pitch = */ 0,
                                  /* host_row_pitch = */ src_stride,
                                  /* host_slice_pitch = */ 0,
                                  src);
    //#ifdef DEDISP_DEBUG
    error |= stream.finish();
    if (error != CL_SUCCESS) {
        return false;
    }
    //#endif
	return true;
}

template </*typename T, */typename U>
bool copy_host_to_device_2d(cl::Buffer& dst, gpu_size_t dst_stride,
                            const U *src, gpu_size_t src_stride,
                            gpu_size_t width_bytes, gpu_size_t height,
                            const cl::CommandQueue& stream = dedisp::device_manager::instance().current_queue()) {
    return copy_host_to_device_2d(dst, 0, dst_stride, src, 0, src_stride, width_bytes, height, stream);
}

template <typename T/*, typename U*/>
bool copy_device_to_host_2d(T *dst, gpu_size_t dst_offset, gpu_size_t dst_stride,
                            cl::Buffer& src, gpu_size_t src_offset, gpu_size_t src_stride,
                            gpu_size_t width_bytes, gpu_size_t height,
                            const cl::CommandQueue& stream = dedisp::device_manager::instance().current_queue()) {
    // TODO: Can't use Async versions unless host memory is pinned!
    cl_int error = 
    stream.enqueueReadBufferRect(src, /* blocking = */ CL_TRUE,
                                 /* buffer_offset = */ {src_offset, 0, 0},
                                 /* host_offset = */ {dst_offset, 0, 0},
                                 /* region = */ {width_bytes, height, 1},
                                 /* buffer_row_pitch = */ src_stride,
                                 /* buffer_slice_pitch = */ 0,
                                 /* host_row_pitch = */ dst_stride,
                                 /* host_slice_pitch = */ 0,
                                 dst);
    //#ifdef DEDISP_DEBUG
    error |= stream.finish();
    if (error != CL_SUCCESS) {
        return false;
    }
    //#endif
	return true;
}

template <typename T/*, typename U*/>
bool copy_device_to_host_2d(T *dst, gpu_size_t dst_stride,
                            cl::Buffer& src, gpu_size_t src_stride,
                            gpu_size_t width_bytes, gpu_size_t height,
                            const cl::CommandQueue& stream = dedisp::device_manager::instance().current_queue()) {
    return copy_device_to_host_2d(dst, 0, dst_stride, src, 0, src_stride, width_bytes, height, stream);
}

/*template <typename T, typename U>*/
bool copy_device_to_device_2d(cl::Buffer& dst, gpu_size_t dst_offset, gpu_size_t dst_stride,
                              cl::Buffer& src, gpu_size_t src_offset, gpu_size_t src_stride,
                              gpu_size_t width_bytes, gpu_size_t height,
                              const cl::CommandQueue& stream = dedisp::device_manager::instance().current_queue()) {
    cl_int error = 
    stream.enqueueCopyBufferRect(src, dst,
                                 /* src_origin = */ {src_offset, 0, 0},
                                 /* dst_origin = */ {dst_offset, 0, 0},
                                 /* region = */ {width_bytes, height, 1},
                                 /* src_row_pitch = */ src_stride,
                                 /* src_slice_pitch = */ 0,
                                 /* dst_row_pitch = */ dst_stride,
                                 /* dst_slice_pitch = */ 0);
    //#ifdef DEDISP_DEBUG
    error |= stream.finish();
    if (error != CL_SUCCESS) {
        return false;
    }
    //#endif
	return true;
}

/*template <typename T, typename U>*/
bool copy_device_to_device_2d(cl::Buffer& dst, gpu_size_t dst_stride,
                              cl::Buffer& src, gpu_size_t src_stride,
                              gpu_size_t width_bytes, gpu_size_t height,
                              const cl::CommandQueue& stream = dedisp::device_manager::instance().current_queue()) {
    return copy_device_to_device_2d(dst, 0, dst_stride, src, 0, src_stride, width_bytes, height, stream);
}
