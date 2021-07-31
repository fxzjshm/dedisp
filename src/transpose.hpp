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
 * This file contains a CUDA implementation of the array transpose operation.
 * 
 * Parts of this file are based on the transpose implementation in the
 * NVIDIA CUDA SDK.
 */

#pragma once

#include <CL/opencl.hpp>
using namespace cl;
#include "utils.dp.hpp"
#include <iostream>
#include <limits>

namespace cuda_specs {
	enum { MAX_GRID_DIMENSION = 65535 };
}

template<typename T>
struct Transpose {
	enum {
//#if __CUDA_ARCH__ < 200
		TILE_DIM     = 32,
//#else
//		TILE_DIM     = 64,
//#endif
		BLOCK_ROWS   = 8//,
		//PAD_MULTIPLE = TILE_DIM
	};
	
	Transpose() {}

    void transpose(cl::Buffer& in, size_t width, size_t height,
                   size_t in_stride, size_t out_stride, cl::Buffer& out,
                   const cl::CommandQueue& stream = dedisp::device_manager::instance().current_queue());
    void transpose(cl::Buffer& in, size_t width, size_t height, cl::Buffer& out,
                   const cl::CommandQueue& stream = dedisp::device_manager::instance().current_queue()) {
            transpose(in, width, height, width, height, out, stream);
	}
private:
	// TODO: These should probably be imported from somewhere else
	template<typename U>
	inline U min(const U& a, const U& b) {
		return a < b ? a : b;
	}
	template<typename U>
	inline U round_up_pow2(const U& a) {
		U r = a-1;
		for( unsigned long i=1; i<=sizeof(U)*8/2; i<<=1 ) r |= r >> i;
		return r+1;
	}
	template<typename U>
	inline U round_down_pow2(const U& a) {
		return round_up_pow2(a+1)/2;
	}
	inline unsigned int log2(unsigned int a) {
		unsigned int r;
		unsigned int shift;
		r =     (a > 0xFFFF) << 4; a >>= r;
		shift = (a > 0xFF  ) << 3; a >>= shift; r |= shift;
		shift = (a > 0xF   ) << 2; a >>= shift; r |= shift;
		shift = (a > 0x3   ) << 1; a >>= shift; r |= shift;
		r |= (a >> 1);
		return r;
	}
	inline unsigned long log2(unsigned long a) {
		unsigned long r;
		unsigned long shift;
		r =     (a > 0xFFFFFFFF) << 5; a >>= r;
		shift = (a > 0xFFFF    ) << 4; a >>= shift; r |= shift;
		shift = (a > 0xFF      ) << 3; a >>= shift; r |= shift;
		shift = (a > 0xF       ) << 2; a >>= shift; r |= shift;
		shift = (a > 0x3       ) << 1; a >>= shift; r |= shift;
		r |= (a >> 1);
		return r;
	}
};

// sync this with gpu_memory.h
// typedef unsigned int gpu_size_t;
typedef cl_uint gpu_size_t;

char transpose_kernel_src[] = {
#include "transpose.cl.xxd.txt"
};


template <typename T>
void Transpose<T>::transpose(cl::Buffer& in,
               size_t width, size_t height,
               size_t in_stride, size_t out_stride,
               cl::Buffer& out,
               const cl::CommandQueue& stream)
{
	// Parameter checks
	// TODO: Implement some sort of error returning!
	if( 0 == width || 0 == height ) return;
	if( 0 == in.get() ) return; //throw std::runtime_error("Transpose: in is NULL");
	if( 0 == out.get() ) return; //throw std::runtime_error("Transpose: out is NULL");
	if( width > in_stride )
		return; //throw std::runtime_error("Transpose: width exceeds in_stride");
	if( height > out_stride )
		return; //throw std::runtime_error("Transpose: height exceeds out_stride");

    
	// Specify thread decomposition (uses up-rounded divisions)
    cl::NDRange tot_block_count((width - 1) / TILE_DIM + 1,
                                (height - 1) / TILE_DIM + 1, 1);

    size_t max_grid_dim = round_down_pow2((size_t)cuda_specs::MAX_GRID_DIMENSION);
	
	// Partition the grid into chunks that the GPU can accept at once
    for (size_t block_y_offset = 0;
         block_y_offset < tot_block_count[1];
         block_y_offset += max_grid_dim) {

        cl_uint block_count[3] = {1, 1, 1};

        // Handle the possibly incomplete final grid
        block_count[1] = min(max_grid_dim,
                             tot_block_count[1] - block_y_offset);

        for (size_t block_x_offset = 0;
             block_x_offset < tot_block_count[0];
             block_x_offset += max_grid_dim) {
        
            // Handle the possibly incomplete final grid
            block_count[0] = min(max_grid_dim,
                                 tot_block_count[0] - block_x_offset);
        
            // Compute the chunked parameters
        	size_t x_offset = block_x_offset * TILE_DIM;
        	size_t y_offset = block_y_offset * TILE_DIM;
        	size_t in_offset = x_offset + y_offset*in_stride;
        	size_t out_offset = y_offset + x_offset*out_stride;
        	size_t w = min(max_grid_dim*TILE_DIM, width-x_offset);
        	size_t h = min(max_grid_dim*TILE_DIM, height-y_offset);
        
            cl::NDRange block(TILE_DIM, BLOCK_ROWS, 1);

            // TODO: Unfortunately there are cases where rounding to a power of 2 becomes
			//       detrimental to performance. Could work out a heuristic.
			//bool round_grid_to_pow2 = false;
			bool round_grid_to_pow2 = true;
			
            auto call_transpose_kernel = [=](bool GRID_IS_POW2, cl::NDRange grid, cl::NDRange block) {
                cl_int error;
                cl::Program program(dedisp::device_manager::instance().current_context(),
                                    transpose_kernel_src, /* build = */ false);
                std::string build_arguments;
                build_arguments += std::string("-DT_TYPE=") + dedisp::get_cl_typename<T>() + " ";
                build_arguments += std::string("-DTILE_DIM=") + std::to_string(TILE_DIM) + " ";
                build_arguments += std::string("-DBLOCK_ROWS=") + std::to_string(BLOCK_ROWS) + " ";
                build_arguments += std::string("-DGRID_IS_POW2=") + std::to_string(int(GRID_IS_POW2)) + " ";
                error = program.build(build_arguments.c_str());
                if (error != CL_SUCCESS) {
                    std::cerr << "Build OpenCL source fail at" << __FILE__ << ":" << __LINE__ << std::endl;
                    auto build_logs = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>();
                    for (auto pair : build_logs) {
                        std::cerr << "Build log of device" << pair.first.getInfo<CL_DEVICE_NAME>() << "is: " << std::endl
                                  << pair.second << std::endl;
                    }
                    
                }
                cl::Kernel kernel(program, "transpose_kernel", &error);
                /*
                 * transpose_kernel(__global const T* in, gpu_size_t in_offset,
                 *                  gpu_size_t width, gpu_size_t height,
                 *                  gpu_size_t in_stride, gpu_size_t out_stride,
                 *                  __global T* out, gpu_size_t out_offset,
                 *                  gpu_size_t block_count_x,
                 *                  gpu_size_t block_count_y,
                 *                  gpu_size_t log2_gridDim_y);
                 */
                kernel.setArg(0, in);
                kernel.setArg(1, (gpu_size_t) in_offset);
                kernel.setArg(2, (gpu_size_t) width);
                kernel.setArg(3, (gpu_size_t) height);
                kernel.setArg(4, (gpu_size_t) in_stride);
                kernel.setArg(5, (gpu_size_t) out_stride);
                kernel.setArg(6, out);
                kernel.setArg(7, (gpu_size_t) out_offset);
                kernel.setArg(8, (gpu_size_t) block_count[0]);
                kernel.setArg(9, (gpu_size_t) block_count[1]);
                kernel.setArg(10, (gpu_size_t) log2(grid[1]));
                cl::NDRange global_size(grid[0] * block[0], grid[1] * block[1], grid[2] * block[2]);
                error = stream.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, block);
            };

			// Dispatch on grid-rounding
			if( round_grid_to_pow2 ) {
                cl::NDRange grid(round_up_pow2(block_count[0]),
                                 round_up_pow2(block_count[1]), 1);
                // Run the CUDA kernel
                call_transpose_kernel(true, grid, block);
            }
			else {
                cl::NDRange grid(block_count[0], block_count[1], 1);
                // Run the CUDA kernel
                call_transpose_kernel(false, grid, block);
            }
			
#ifndef NDEBUG
            cl_int error = stream.finish();
			if( error != CL_SUCCESS ) {
				/*
				throw std::runtime_error(
					std::string("Transpose: CUDA error in kernel: ") +
					cudaGetErrorString(error));
				*/
			}
#endif
		}
	}
}
