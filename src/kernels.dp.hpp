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
  This file contains the important stuff like the CUDA kernel and physical
    equations.
*/

#pragma once

#include <boost/compute.hpp>
namespace bc = boost::compute;

#include <vector> // For generate_dm_list
#include <cmath>
#include <algorithm>

#include "utils.dp.hpp"

// Kernel tuning parameters
#define DEDISP_BLOCK_SIZE       256
#define DEDISP_BLOCK_SAMPS      8
#define DEDISP_SAMPS_PER_THREAD 2 // 4 is better for Fermi?

// dedisp_float c_delay_table[DEDISP_MAX_NCHANS];
// dedisp_bool  c_killmask[DEDISP_MAX_NCHANS];
bc::buffer c_delay_table;
bc::buffer c_killmask;

template<int NBITS, typename T=unsigned int>
struct max_value {
	static const T value = (((unsigned)1<<(NBITS-1))-1)*2+1;
};

void generate_delay_table(dedisp_float* h_delay_table, dedisp_size nchans,
						  dedisp_float dt, dedisp_float f0, dedisp_float df)
{
	for( dedisp_size c=0; c<nchans; ++c ) {
		dedisp_float a = 1.f / (f0+c*df);
		dedisp_float b = 1.f / f0;
		// Note: To higher precision, the constant is 4.148741601e3
		h_delay_table[c] = 4.15e3/dt * (a*a - b*b);
	}
}

void generate_dm_list(std::vector<dedisp_float>& dm_table,
					  dedisp_float dm_start, dedisp_float dm_end,
					  double dt, double ti, double f0, double df,
					  dedisp_size nchans, double tol)
{
	// Note: This algorithm originates from Lina Levin
	// Note: Computation done in double precision to match MB's code
	
	dt *= 1e6;
	double f    = (f0 + ((nchans/2) - 0.5) * df) * 1e-3;
	double tol2 = tol*tol;
	double a    = 8.3 * df / (f*f*f);
	double a2   = a*a;
	double b2   = a2 * (double)(nchans*nchans / 16.0);
	double c    = (dt*dt + ti*ti) * (tol2 - 1.0);
	
	dm_table.push_back(dm_start);
	while( dm_table.back() < dm_end ) {
		double prev     = dm_table.back();
		double prev2    = prev*prev;
		double k        = c + tol2*a2*prev2;
		double dm = ((b2*prev + sqrt(-a2*b2*prev2 + (a2+b2)*k)) / (a2+b2));
		dm_table.push_back(dm);
	}
}

char dedisperse_kernel_src[] = {
#include "dedisperse.cl.xxd.txt"
};

bool dedisperse(bc::buffer       d_in,
                dedisp_size      d_in_offset,
                dedisp_size      in_stride,
                dedisp_size      nsamps,
                dedisp_size      in_nbits,
                dedisp_size      nchans,
                dedisp_size      chan_stride,
                const bc::buffer d_dm_list,
                dedisp_size      d_dm_list_offset,
                dedisp_size      dm_count,
                dedisp_size      dm_stride,
                bc::buffer       d_out,
                dedisp_size      d_out_offset,
                dedisp_size      out_stride,
                dedisp_size      out_nbits,
                dedisp_size      batch_size,
                dedisp_size      batch_in_stride,
                dedisp_size      batch_dm_stride,
                dedisp_size      batch_chan_stride,
                dedisp_size      batch_out_stride) {
	enum {
		BITS_PER_BYTE            = 8,
		BYTES_PER_WORD           = sizeof(dedisp_word) / sizeof(dedisp_byte),
		BLOCK_DIM_X              = DEDISP_BLOCK_SAMPS,
		BLOCK_DIM_Y              = DEDISP_BLOCK_SIZE / DEDISP_BLOCK_SAMPS,
		MAX_CUDA_GRID_SIZE_X     = 65535,
		MAX_CUDA_GRID_SIZE_Y     = 65535,
		MAX_CUDA_1D_TEXTURE_SIZE = (1<<27)
	};
	
	// Initialise texture memory if necessary
    // In OpenCL, not necessary; and some devices do not support texture
	
	// Define thread decomposition
	// Note: Block dimensions x and y represent time samples and DMs respectively
    bc::extents<3> block = {BLOCK_DIM_X, BLOCK_DIM_Y, 1};
    // Note: Grid dimension x represents time samples. Dimension y represents
	//         DMs and batch jobs flattened together.
	
	// Divide and round up
    dedisp_size nsamp_blocks =
        (nsamps - 1) / ((dedisp_size)DEDISP_SAMPS_PER_THREAD * block[0]) + 1;
    dedisp_size ndm_blocks = (dm_count - 1) / (dedisp_size)block[1] + 1;

    // Constrain the grid size to the maximum allowed
	// TODO: Consider cropping the batch size dimension instead and looping over it
	//         inside the kernel
    ndm_blocks = std::min((unsigned int)ndm_blocks,
                          (unsigned int)(MAX_CUDA_GRID_SIZE_Y / batch_size));

    // Note: We combine the DM and batch dimensions into one
    bc::extents<3> grid = {nsamp_blocks, ndm_blocks * batch_size, 1};

    // Divide and round up
	dedisp_size nsamps_reduced = (nsamps - 1) / DEDISP_SAMPS_PER_THREAD + 1;

    bc::command_queue stream = bc::system::default_queue();

    bc::context context = bc::system::default_context();
    // bc::buffer d_c_delay_table(context, (size_t)DEDISP_MAX_NCHANS, CL_MEM_COPY_HOST_PTR, c_delay_table);
    // bc::buffer d_c_killmask(context, (size_t)DEDISP_MAX_NCHANS, CL_MEM_COPY_HOST_PTR, c_killmask);

    // Execute the kernel
    auto DEDISP_CALL_KERNEL = [&](int NBITS) {
        cl_int error;
        bc::program program = bc::program::create_with_source(dedisperse_kernel_src, bc::system::default_context());
        std::string build_arguments = dedisp::type_define_arguments;
// macros need to be defined when compile:
//     DEDISP_WORD_TYPE, DEDISP_SIZE_TYPE, DEDISP_FLOAT_TYPE, DEDISP_BYTE_TYPE, DEDISP_BOOL_TYPE
//     int IN_NBITS, int SAMPS_PER_THREAD, int BLOCK_DIM_X, int BLOCK_DIM_Y
        build_arguments += std::string("-DIN_NBITS=") + std::to_string(NBITS) + " ";
        build_arguments += std::string("-DSAMPS_PER_THREAD=") + std::to_string(DEDISP_SAMPS_PER_THREAD) + " ";
        build_arguments += std::string("-DBLOCK_DIM_X=") + std::to_string(BLOCK_DIM_X) + " ";
        build_arguments += std::string("-DBLOCK_DIM_Y=") + std::to_string(BLOCK_DIM_Y) + " ";
        try {
            program.build(build_arguments.c_str());
        } catch(bc::opencl_error error) {
            std::cerr << "Build OpenCL source fail at" << __FILE__ << ":" << __LINE__ << std::endl;
            std::cerr << "Build log is: " << std::endl
                      << program.build_log() << std::endl;
        }
        bc::kernel kernel = program.create_kernel("dedisperse_kernel");

        /*
__kernel void dedisperse_kernel(
              __global const dedisp_word*  d_in,
                       dedisp_size         d_in_offset,
                       dedisp_size         nsamps,
                       dedisp_size         nsamps_reduced,
                       dedisp_size         nsamp_blocks,
                       dedisp_size         stride,
                       dedisp_size         dm_count,
                       dedisp_size         dm_stride,
                       dedisp_size         ndm_blocks,
                       dedisp_size         nchans,
                       dedisp_size         chan_stride,
              __global dedisp_byte*        d_out,
                       dedisp_size         d_out_offset,
                       dedisp_size         out_nbits,
                       dedisp_size         out_stride,
              __global const dedisp_float* d_dm_list,
                       dedisp_size         d_dm_list_offset,
                       dedisp_size         batch_in_stride,
                       dedisp_size         batch_dm_stride,
                       dedisp_size         batch_chan_stride,
                       dedisp_size         batch_out_stride,
            __constant dedisp_float*       c_delay_table,
            __constant dedisp_bool*        c_killmask);
        */
        kernel.set_arg(0, d_in);
        kernel.set_arg(1, d_in_offset);
        kernel.set_arg(2, nsamps);
        kernel.set_arg(3, nsamps_reduced);
        kernel.set_arg(4, nsamp_blocks);
        kernel.set_arg(5, in_stride);
        kernel.set_arg(6, dm_count);
        kernel.set_arg(7, dm_stride);
        kernel.set_arg(8, ndm_blocks);
        kernel.set_arg(9, nchans);
        kernel.set_arg(10, chan_stride);
        kernel.set_arg(11, d_out);
        kernel.set_arg(12, d_out_offset);
        kernel.set_arg(13, out_nbits);
        kernel.set_arg(14, out_stride);
        kernel.set_arg(15, d_dm_list);
        kernel.set_arg(16, d_dm_list_offset);
        kernel.set_arg(17, batch_in_stride);
        kernel.set_arg(18, batch_dm_stride);
        kernel.set_arg(19, batch_chan_stride);
        kernel.set_arg(20, batch_out_stride);
        kernel.set_arg(21, c_delay_table);
        kernel.set_arg(22, c_killmask);

        bc::extents<3> global_size = bc::dim(grid[0] * block[0], grid[1] * block[1], grid[2] * block[2]);
        stream.enqueue_nd_range_kernel(kernel, 3, nullptr, global_size.data(), block.data());
    };
    // Note: Here we dispatch dynamically on nbits for supported values
    switch( in_nbits ) {
        case 1:  DEDISP_CALL_KERNEL(1);  break;
        case 2:  DEDISP_CALL_KERNEL(2);  break;
        case 4:  DEDISP_CALL_KERNEL(4);  break;
        case 8:  DEDISP_CALL_KERNEL(8);  break;
        case 16: DEDISP_CALL_KERNEL(16); break;
        case 32: DEDISP_CALL_KERNEL(32); break;
        default: /* should never be reached */ break;
    }
#ifdef DEDISP_DEBUG
    stream.finish();  
#endif
    return true;
}

const char scrunch_x2_src[] = {
#include "scrunch_x2.cl.xxd.txt"
};

// Reduces the time resolution by 2x
dedisp_error scrunch_x2(bc::buffer  d_in, dedisp_size d_in_offset,
                        dedisp_size nsamps,
                        dedisp_size nchan_words,
                        dedisp_size nbits,
                        bc::buffer  d_out, dedisp_size d_out_offset)
{
    dedisp_size out_nsamps = nsamps / 2;
	dedisp_size out_count  = out_nsamps * nchan_words;

    bc::program program = bc::program::create_with_source(scrunch_x2_src, bc::system::default_context());
    std::string build_arguments = dedisp::type_define_arguments;
    program.build(build_arguments.c_str());
    bc::kernel kernel = program.create_kernel("scrunch_x2_kernel");
    /* __kernel void scrunch_x2_kernel(__global WordType* in, dedisp_size in_offset __global dedisp_word* outs, dedisp_size out_offset, int nbits, unsigned int in_nsamps); */
    kernel.set_arg(0, d_in);
    kernel.set_arg(1, d_in_offset);
    kernel.set_arg(2, d_out);
    kernel.set_arg(3, d_out_offset);
    kernel.set_arg(4, (cl_int) nbits);
    kernel.set_arg(5, (cl_uint) nsamps);
    bc::command_queue queue = bc::system::default_queue();
    queue.enqueue_1d_range_kernel(kernel, 0, out_count, 0);
    queue.finish();

    return DEDISP_NO_ERROR;
}

dedisp_float get_smearing(dedisp_float dt, dedisp_float pulse_width,
                          dedisp_float f0, dedisp_size nchans, dedisp_float df,
                          dedisp_float DM, dedisp_float deltaDM) {
	dedisp_float W         = pulse_width;
	dedisp_float BW        = nchans * abs(df);
	dedisp_float fc        = f0 - BW/2;
	dedisp_float inv_fc3   = 1./(fc*fc*fc);
	dedisp_float t_DM      = 8.3*BW*DM*inv_fc3;
	dedisp_float t_deltaDM = 8.3/4*BW*nchans*deltaDM*inv_fc3;
	dedisp_float t_smear   = sqrt(dt*dt + W*W + t_DM*t_DM + t_deltaDM*t_deltaDM);
	return t_smear;
}

dedisp_error generate_scrunch_list(dedisp_size* scrunch_list,
                                   dedisp_size dm_count,
                                   dedisp_float dt0,
                                   const dedisp_float* dm_list,
                                   dedisp_size nchans,
                                   dedisp_float f0, dedisp_float df,
                                   dedisp_float pulse_width,
                                   dedisp_float tol) {
	// Note: This algorithm always starts with no scrunching and is only
	//         able to 'adapt' the scrunching by doubling in any step.
	// TODO: To improve this it would be nice to allow scrunch_list[0] > 1.
	//         This would probably require changing the output nsamps
	//           according to the mininum scrunch.
	
	scrunch_list[0] = 1;
	for( dedisp_size d=1; d<dm_count; ++d ) {
		dedisp_float dm = dm_list[d];
		dedisp_float delta_dm = dm - dm_list[d-1];
		
		dedisp_float smearing = get_smearing(scrunch_list[d-1] * dt0,
		                                     pulse_width*1e-6,
		                                     f0, nchans, df,
		                                     dm, delta_dm);
		dedisp_float smearing2 = get_smearing(scrunch_list[d-1] * 2 * dt0,
		                                      pulse_width*1e-6,
		                                      f0, nchans, df,
		                                      dm, delta_dm);
		if( smearing2 / smearing < tol ) {
			scrunch_list[d] = scrunch_list[d-1] * 2;
		}
		else {
			scrunch_list[d] = scrunch_list[d-1];
		}
	}
	
	return DEDISP_NO_ERROR;
}

char unpack_kernel_src[] = {
#include "unpack.cl.xxd.txt"
};

dedisp_error unpack(bc::buffer d_transposed,
                    dedisp_size nsamps, dedisp_size nchan_words,
                    bc::buffer d_unpacked,
                    dedisp_size in_nbits, dedisp_size out_nbits)
{
    dedisp_size expansion = out_nbits / in_nbits;
	dedisp_size in_count  = nsamps * nchan_words;
	dedisp_size out_count = in_count * expansion;

    bc::program program = bc::program::create_with_source(unpack_kernel_src, bc::system::default_context());
    std::string build_arguments = dedisp::type_define_arguments;
    program.build(build_arguments.c_str());
    bc::kernel kernel = program.create_kernel("unpack_kernel");
    /* __kernel void unpack_kernel(__global WordType* in, __global dedisp_word* out, int nsamps, int in_nbits, int out_nbits); */
    kernel.set_arg(0, d_transposed);
    kernel.set_arg(1, d_unpacked);
    kernel.set_arg(2, (cl_int) nsamps);
    kernel.set_arg(3, (cl_int) in_nbits);
    kernel.set_arg(4, (cl_int) out_nbits);
    bc::command_queue queue = bc::system::default_queue();
    queue.enqueue_1d_range_kernel(kernel, 0, out_count, 0);
    queue.finish();

    return DEDISP_NO_ERROR;
}
