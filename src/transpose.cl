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
  This file contains functions transpose_kernel
*/

// macros need to be defined when compile:
// T_TYPE, TILE_DIM, BLOCK_ROWS, GRID_IS_POW2

// #define USE_TEST_DEFINES
// test use only
#ifdef USE_TEST_DEFINES 
#define T_TYPE uint
#define TILE_DIM 32
#define BLOCK_ROWS 8
#define GRID_IS_POW2 1
#endif

typedef uint /* = cl_uint */ gpu_size_t;
typedef T_TYPE T;

__kernel void transpose_kernel(__global const T* in, gpu_size_t in_offset,
					           gpu_size_t width, gpu_size_t height,
					           gpu_size_t in_stride, gpu_size_t out_stride,
					           __global T* out, gpu_size_t out_offset,
					           gpu_size_t block_count_x,
					           gpu_size_t block_count_y,
					           gpu_size_t log2_gridDim_y)
{
    __local T tile[TILE_DIM][TILE_DIM+1];

    gpu_size_t blockIdx_x, blockIdx_y;
    uint3 blockIdx = (uint3) ((uint) get_group_id(0), (uint) get_group_id(1), (uint) get_group_id(2));
    uint3 gridDim = (uint3) ((uint) get_num_groups(0), (uint) get_num_groups(1), (uint) get_num_groups(2));
    uint3 threadIdx = (uint3) ((uint) get_local_id(0), (uint) get_local_id(1), (uint) get_local_id(2));
	
	// Do diagonal index reordering to avoid partition camping in device memory
	if( width == height ) {
        blockIdx_y = blockIdx.x;
        if( !GRID_IS_POW2 ) {
            blockIdx_x = (blockIdx.x+blockIdx.y) % gridDim.x;
        }
		else {
            blockIdx_x = (blockIdx.x+blockIdx.y) & (gridDim.x-1);
        }
	}
	else {
        gpu_size_t bid = blockIdx.x + gridDim.x*blockIdx.y;
        if( !GRID_IS_POW2 ) {
            blockIdx_y = bid % gridDim.y;
            blockIdx_x = ((bid / gridDim.y) + blockIdx_y) % gridDim.x;
        }
		else {
			blockIdx_y = bid & (gridDim.y-1);
			blockIdx_x = ((bid >> log2_gridDim_y) + blockIdx_y) & (gridDim.x-1);
        }
	}
	
	// Cull excess blocks (there may be many if we round up to a power of 2)
	if( blockIdx_x >= block_count_x ||
		blockIdx_y >= block_count_y ) {
		return;
	}

    gpu_size_t index_in_x = blockIdx_x * TILE_DIM + threadIdx.x;
    gpu_size_t index_in_y = blockIdx_y * TILE_DIM + threadIdx.y;
    gpu_size_t index_in = index_in_x + (index_in_y)*in_stride;
	
#pragma unroll
	for( gpu_size_t i=0; i<TILE_DIM; i+=BLOCK_ROWS ) {
		// TODO: Is it possible to cull some excess threads early?
		if( index_in_x < width && index_in_y+i < height )
            tile[threadIdx.y+i][threadIdx.x] = in[in_offset+index_in+i*in_stride];
    }

    barrier(CLK_LOCAL_MEM_FENCE); // TODO check whether we need CLK_GLOBAL_MEM_FENCE

    gpu_size_t index_out_x = blockIdx_y * TILE_DIM + threadIdx.x;
    // Avoid excess threads
	if( index_out_x >= height ) return;
    gpu_size_t index_out_y = blockIdx_x * TILE_DIM + threadIdx.y;
    gpu_size_t index_out = index_out_x + (index_out_y)*out_stride;
	
#pragma unroll
	for( gpu_size_t i=0; i<TILE_DIM; i+=BLOCK_ROWS ) {
		// Avoid excess threads
		if( index_out_y+i < width ) {
            out[out_offset+index_out+i*out_stride] = tile[threadIdx.x][threadIdx.y+i];
        }
	}
}
