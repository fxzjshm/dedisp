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

#include <boost/compute.hpp>
namespace bc = boost::compute;

#include "utils.dp.hpp"
// typedef unsigned int gpu_size_t;

template <typename T>
bool malloc_device(bc::buffer &addr, gpu_size_t count) {
    try {
        addr = bc::buffer(bc::system::default_context(), count * sizeof(T), CL_MEM_READ_WRITE, nullptr);
    } catch(...) {
        return false;
    }
    return true;
}
template<typename T>
void free_device(bc::buffer &addr) {
        std::move(addr);
}
template <typename T>
bool copy_host_to_device(bc::buffer& dst, const T* src, gpu_size_t count,
                         bc::command_queue& stream = bc::system::default_queue()) {
    try {
        // TODO: Can't use Async versions unless host memory is pinned!
        // TODO: Passing a device pointer as src causes this to segfault
        stream.enqueue_write_buffer(dst, CL_TRUE, 0, count * sizeof(T), src);
        stream.finish();
    } catch (...) {
        return false;
    }
    return true;
}
template <typename T>
bool copy_device_to_host(T *dst, bc::buffer& src, gpu_size_t count,
                         bc::command_queue& stream = bc::system::default_queue()) {
    try {
        // TODO: Can't use Async versions unless host memory is pinned!
        stream.enqueue_read_buffer(src, 0, count * sizeof(T), dst);
        stream.finish();
    } catch (...) {
        return false;
    }
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
bool copy_host_to_device_2d(bc::buffer& dst, gpu_size_t dst_offset, gpu_size_t dst_stride,
                            const U *src, gpu_size_t src_offset, gpu_size_t src_stride,
                            gpu_size_t width_bytes, gpu_size_t height,
                            bc::command_queue& stream = bc::system::default_queue()) {
    try {
        // TODO: Can't use Async versions unless host memory is pinned!
        stream.enqueue_write_buffer_rect(dst,
                                         /* buffer_offset = */ bc::dim(dst_offset, 0, 0).data(),
                                         /* host_offset = */ bc::dim(src_stride, 0, 0).data(),
                                         /* region = */ bc::dim(width_bytes, height, 1).data(),
                                         /* buffer_row_pitch = */ dst_stride,
                                         /* buffer_slice_pitch = */ 0,
                                         /* host_row_pitch = */ src_stride,
                                         /* host_slice_pitch = */ 0,
                                         const_cast<U*>(src));
        stream.finish();
    } catch (...) {
        return false;
    }
    return true;
}

template </*typename T, */typename U>
bool copy_host_to_device_2d(bc::buffer& dst, gpu_size_t dst_stride,
                            const U *src, gpu_size_t src_stride,
                            gpu_size_t width_bytes, gpu_size_t height,
                            bc::command_queue& stream = bc::system::default_queue()) {
    return copy_host_to_device_2d(dst, 0, dst_stride, src, 0, src_stride, width_bytes, height, stream);
}

template <typename T/*, typename U*/>
bool copy_device_to_host_2d(T *dst, gpu_size_t dst_offset, gpu_size_t dst_stride,
                            bc::buffer& src, gpu_size_t src_offset, gpu_size_t src_stride,
                            gpu_size_t width_bytes, gpu_size_t height,
                            bc::command_queue& stream = bc::system::default_queue()) {
    try {
        // TODO: Can't use Async versions unless host memory is pinned!
        stream.enqueue_read_buffer_rect(src,
                                        /* buffer_offset = */ bc::dim(src_offset, 0, 0).data(),
                                        /* host_offset = */ bc::dim(dst_offset, 0, 0).data(),
                                        /* region = */ bc::dim(width_bytes, height, 1).data(),
                                        /* buffer_row_pitch = */ src_stride,
                                        /* buffer_slice_pitch = */ 0,
                                        /* host_row_pitch = */ dst_stride,
                                        /* host_slice_pitch = */ 0,
                                        dst);
        stream.finish();
    } catch (...) {
        return false;
    }
    return true;
}

template <typename T/*, typename U*/>
bool copy_device_to_host_2d(T *dst, gpu_size_t dst_stride,
                            bc::buffer& src, gpu_size_t src_stride,
                            gpu_size_t width_bytes, gpu_size_t height,
                            bc::command_queue& stream = bc::system::default_queue()) {
    return copy_device_to_host_2d(dst, 0, dst_stride, src, 0, src_stride, width_bytes, height, stream);
}

/*template <typename T, typename U>*/
bool copy_device_to_device_2d(bc::buffer& dst, gpu_size_t dst_offset, gpu_size_t dst_stride,
                              bc::buffer& src, gpu_size_t src_offset, gpu_size_t src_stride,
                              gpu_size_t width_bytes, gpu_size_t height,
                              bc::command_queue& stream = bc::system::default_queue()) {
    try {
        stream.enqueue_copy_buffer_rect(src, dst,
                                        bc::dim(src_offset, 0, 0).data(),
                                        bc::dim(dst_offset, 0, 0).data(),
                                        bc::dim(width_bytes, height, 1).data(),
                                        /* src_row_pitch = */ src_stride,
                                        /* src_slice_pitch = */ 0,
                                        /* dst_row_pitch = */ dst_stride,
                                        /* dst_slice_pitch = */ 0);
        stream.finish();
    } catch (...) {
        return false;
    }
    return true;
}

/*template <typename T, typename U>*/
bool copy_device_to_device_2d(bc::buffer& dst, gpu_size_t dst_stride,
                              bc::buffer& src, gpu_size_t src_stride,
                              gpu_size_t width_bytes, gpu_size_t height,
                              bc::command_queue& stream = bc::system::default_queue()) {
    return copy_device_to_device_2d(dst, 0, dst_stride, src, 0, src_stride, width_bytes, height, stream);
}
