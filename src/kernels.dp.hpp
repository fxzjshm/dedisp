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

#include <CL/opencl.hpp>
#include <boost/compute.hpp>
using namespace cl;

#include <vector> // For generate_dm_list
#include <cmath>
#include <algorithm>

#include "utils.dp.hpp"

// Kernel tuning parameters
#define DEDISP_BLOCK_SIZE       256
#define DEDISP_BLOCK_SAMPS      8
#define DEDISP_SAMPS_PER_THREAD 2 // 4 is better for Fermi?

dedisp_float c_delay_table[DEDISP_MAX_NCHANS];
dedisp_bool  c_killmask[DEDISP_MAX_NCHANS];

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

bool dedisperse(cl::Buffer       d_in,
                dedisp_size      d_in_offset,
                dedisp_size      in_stride,
                dedisp_size      nsamps,
                dedisp_size      in_nbits,
                dedisp_size      nchans,
                dedisp_size      chan_stride,
                const cl::Buffer d_dm_list,
                dedisp_size      d_dm_list_offset,
                dedisp_size      dm_count,
                dedisp_size      dm_stride,
                cl::Buffer       d_out,
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
    cl::NDRange block(BLOCK_DIM_X, BLOCK_DIM_Y, 1);
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
    cl::NDRange grid(nsamp_blocks, ndm_blocks * batch_size, 1);

    // Divide and round up
	dedisp_size nsamps_reduced = (nsamps - 1) / DEDISP_SAMPS_PER_THREAD + 1;

    cl::CommandQueue stream = dedisp::device_manager::instance().current_device().default_queue();

    cl::Context context = dedisp::device_manager::instance().current_context();
    cl::Buffer d_c_delay_table(context, CL_MEM_COPY_HOST_PTR, (size_t)DEDISP_MAX_NCHANS, c_delay_table);
    cl::Buffer d_c_killmask(context, CL_MEM_COPY_HOST_PTR, (size_t)DEDISP_MAX_NCHANS, c_killmask);

    // Execute the kernel
    auto DEDISP_CALL_KERNEL = [=](int NBITS) {
        cl_int error;
        cl::Program program(dedisp::device_manager::instance().current_context(),
                            dedisperse_kernel_src, /* build = */ false);
        std::string build_arguments = dedisp::type_define_arguments;
// macros need to be defined when compile:
//     DEDISP_WORD_TYPE, DEDISP_SIZE_TYPE, DEDISP_FLOAT_TYPE, DEDISP_BYTE_TYPE, DEDISP_BOOL_TYPE
//     int IN_NBITS, int SAMPS_PER_THREAD, int BLOCK_DIM_X, int BLOCK_DIM_Y
        build_arguments += std::string("-DIN_NBITS=") + std::to_string(NBITS) + " ";
        build_arguments += std::string("-DSAMPS_PER_THREAD=") + std::to_string(DEDISP_SAMPS_PER_THREAD) + " ";
        build_arguments += std::string("-DBLOCK_DIM_X=") + std::to_string(BLOCK_DIM_X) + " ";
        build_arguments += std::string("-DBLOCK_DIM_Y=") + std::to_string(BLOCK_DIM_Y) + " ";
        error = program.build(build_arguments.c_str());
        if (error != CL_SUCCESS) {
            std::cerr << "Build OpenCL source fail at" << __FILE__ << ":" << __LINE__ << std::endl;
            auto build_logs = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>();
            for (auto pair : build_logs) {
                std::cerr << "Build log of device" << pair.first.getInfo<CL_DEVICE_NAME>() << "is: " << std::endl
                          << pair.second << std::endl;
            }
        }
        cl::Kernel kernel(program, "dedisperse_kernel", &error);

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
        kernel.setArg(0, d_in);
        kernel.setArg(1, d_in_offset);
        kernel.setArg(2, nsamps);
        kernel.setArg(3, nsamps_reduced);
        kernel.setArg(4, nsamp_blocks);
        kernel.setArg(5, in_stride);
        kernel.setArg(6, dm_count);
        kernel.setArg(7, dm_stride);
        kernel.setArg(8, ndm_blocks);
        kernel.setArg(9, nchans);
        kernel.setArg(10, chan_stride);
        kernel.setArg(11, d_out);
        kernel.setArg(12, d_out_offset);
        kernel.setArg(13, out_nbits);
        kernel.setArg(14, out_stride);
        kernel.setArg(15, d_dm_list);
        kernel.setArg(16, d_dm_list_offset);
        kernel.setArg(17, batch_in_stride);
        kernel.setArg(18, batch_dm_stride);
        kernel.setArg(19, batch_chan_stride);
        kernel.setArg(20, batch_out_stride);
        kernel.setArg(21, d_c_delay_table);
        kernel.setArg(22, d_c_killmask);
        cl::NDRange global_size(grid[0] * block[0], grid[1] * block[1], grid[2] * block[2]);
        error = stream.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, block);
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

#undef DEDISP_CALL_KERNEL
		
	// Check for kernel errors
#ifdef DEDISP_DEBUG
	cl_int error = stream.finish();
	if( error != CL_SUCCESS ) {
		return false;
	}
#endif // DEDISP_DEBUG
	
	return true;
}

const char scrunch_x2_src[] = {
#include "scrunch_x2.cl.xxd.txt"
};

// Reduces the time resolution by 2x
dedisp_error scrunch_x2(cl::Buffer  d_in, dedisp_size d_in_offset,
                        dedisp_size nsamps,
                        dedisp_size nchan_words,
                        dedisp_size nbits,
                        cl::Buffer  d_out, dedisp_size d_out_offset)
{
    dedisp_size out_nsamps = nsamps / 2;
	dedisp_size out_count  = out_nsamps * nchan_words;

    cl::Program program(dedisp::device_manager::instance().current_context(), scrunch_x2_src);
    std::string build_arguments = dedisp::type_define_arguments;
    program.build(build_arguments.c_str());
    cl::Kernel kernel(program, "scrunch_x2_kernel");
    /* __kernel void scrunch_x2_kernel(__global WordType* in, dedisp_size in_offset __global dedisp_word* outs, dedisp_size out_offset, int nbits, unsigned int in_nsamps); */
    kernel.setArg(0, d_in);
    kernel.setArg(1, d_in_offset);
    kernel.setArg(2, d_out);
    kernel.setArg(3, d_out_offset);
    kernel.setArg(4, (cl_int) nbits);
    kernel.setArg(5, (cl_uint) nsamps);
    cl::CommandQueue queue = dedisp::device_manager::instance().current_queue();
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(out_count));
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

dedisp_error unpack(cl::Buffer d_transposed,
                    dedisp_size nsamps, dedisp_size nchan_words,
                    cl::Buffer d_unpacked,
                    dedisp_size in_nbits, dedisp_size out_nbits)
{
    dedisp_size expansion = out_nbits / in_nbits;
	dedisp_size in_count  = nsamps * nchan_words;
	dedisp_size out_count = in_count * expansion;

    cl::Program program(dedisp::device_manager::instance().current_context(), unpack_kernel_src);
    string build_arguments = dedisp::type_define_arguments;
    program.build(build_arguments.c_str());
    cl::Kernel kernel(program, "unpack_kernel");
    /* __kernel void unpack_kernel(__global WordType* in, __global dedisp_word* out, int nsamps, int in_nbits, int out_nbits); */
    kernel.setArg(0, d_transposed);
    kernel.setArg(1, d_unpacked);
    kernel.setArg(2, (cl_int) nsamps);
    kernel.setArg(3, (cl_int) in_nbits);
    kernel.setArg(4, (cl_int) out_nbits);
    cl::CommandQueue queue = dedisp::device_manager::instance().current_queue();
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(out_count));
    queue.finish();

    return DEDISP_NO_ERROR;
}
