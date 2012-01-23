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

#include <vector> // For generate_dm_list

// Kernel tuning parameters
#define DEDISP_BLOCK_SIZE       256 // 256 best for direct, 128 best for sub-band
#define DEDISP_BLOCK_SAMPS      8   // 8 best for direct, 16 best for sub-band
#define DEDISP_SAMPS_PER_THREAD 4

__constant__ dedisp_float c_delay_table[DEDISP_MAX_NCHANS];
__constant__ dedisp_bool  c_killmask[DEDISP_MAX_NCHANS];

// Texture reference for input data
texture<dedisp_word, 1, cudaReadModeElementType> t_in;

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
	while( dm_table.back() <= dm_end ) {
		double prev     = dm_table.back();
		double prev2    = prev*prev;
		double k        = c + tol2*a2*prev2;
		double dm = ((b2*prev + sqrt(-a2*b2*prev2 + (a2+b2)*k)) / (a2+b2));
		dm_table.push_back(dm);
	}
}

template<int IN_NBITS, typename T, typename SumType>
inline __host__ __device__
T scale_output(SumType sum, dedisp_size nchans) {
	enum { BITS_PER_BYTE = 8 };
	// This emulates dedisperse_all, but is specific to 8-bit output
	// Note: This also breaks the sub-band algorithm
	//return (dedisp_word)(((unsigned int)sum >> 4) - 128 - 128);
	
	// This uses the full range of the output bits
	return (T)((float)sum / ((float)nchans * max_value<IN_NBITS>::value)
			   * max_value<sizeof(T)*BITS_PER_BYTE>::value );
}

template<int NBITS, typename T>
inline __host__ __device__
T extract_subword(T value, int idx) {
	enum { MASK = max_value<NBITS,T>::value };
	return (value >> (idx*NBITS)) & MASK;
}

// Summation type metafunction
template<int IN_NBITS> struct SumType { typedef dedisp_word type; };
// Note: For 32-bit input, we must accumulate using a larger data type
template<> struct SumType<32> { typedef unsigned long long type; };

// Note: This assumes consecutive input words are consecutive times,
//         but that consecutive subwords are consecutive channels.
//       E.g., Words bracketed: (t0c0,t0c1,t0c2,t0c3), (t1c0,t1c1,t1c2,t1c3),...
// Note: out_stride should be in units of samples
template<int IN_NBITS, int SAMPS_PER_THREAD,
		 int BLOCK_DIM_X, int BLOCK_DIM_Y,
		 bool USE_TEXTURE_MEM>
__global__
void dedisperse_kernel(const dedisp_word*  d_in,
                       dedisp_size         nsamps,
                       dedisp_size         nsamps_reduced,
                       dedisp_size         nsamp_blocks,
                       dedisp_size         stride,
                       dedisp_size         dm_count,
                       dedisp_size         dm_stride,
                       dedisp_size         ndm_blocks,
                       dedisp_size         nchans,
                       dedisp_size         chan_stride,
                       dedisp_byte*        d_out,
                       dedisp_size         out_nbits,
                       dedisp_size         out_stride,
                       const dedisp_float* d_dm_list,
                       dedisp_size         batch_in_stride,
                       dedisp_size         batch_dm_stride,
                       dedisp_size         batch_chan_stride,
                       dedisp_size         batch_out_stride)
{
	// Compute compile-time constants
	enum {
		BITS_PER_BYTE  = 8,
		CHANS_PER_WORD = sizeof(dedisp_word) * BITS_PER_BYTE / IN_NBITS
	};
	
	// Compute the thread decomposition
	dedisp_size samp_block    = blockIdx.x;
	dedisp_size dm_block      = blockIdx.y % ndm_blocks;
	dedisp_size batch_block   = blockIdx.y / ndm_blocks;
	
	dedisp_size samp_idx      = samp_block   * BLOCK_DIM_X + threadIdx.x;
	dedisp_size dm_idx        = dm_block     * BLOCK_DIM_Y + threadIdx.y;
	dedisp_size batch_idx     = batch_block;
	dedisp_size nsamp_threads = nsamp_blocks * BLOCK_DIM_X;
	
	dedisp_size ndm_threads   = ndm_blocks * BLOCK_DIM_Y;
	
	// Iterate over grids of DMs
	for( ; dm_idx < dm_count; dm_idx += ndm_threads ) {
	
	// Look up the dispersion measure
	// Note: The dm_stride and batch_dm_stride params are only used for the
	//         sub-band method.
	dedisp_float dm = d_dm_list[dm_idx*dm_stride + batch_idx*batch_dm_stride];
	
	// Loop over samples
	for( ; samp_idx < nsamps_reduced; samp_idx += nsamp_threads ) {
		typedef typename SumType<IN_NBITS>::type sum_type;
		sum_type sum[SAMPS_PER_THREAD];
		
        #pragma unroll
		for( dedisp_size s=0; s<SAMPS_PER_THREAD; ++s ) {
			sum[s] = 0;
		}
		
		// Loop over channel words
		for( dedisp_size chan_word=0; chan_word<nchans;
		     chan_word+=CHANS_PER_WORD ) {
			// Pre-compute the memory offset
			dedisp_size offset = 
				samp_idx*SAMPS_PER_THREAD
				+ chan_word/CHANS_PER_WORD * stride
				+ batch_idx * batch_in_stride;
			
			// Loop over channel subwords
			for( dedisp_size chan_sub=0; chan_sub<CHANS_PER_WORD; ++chan_sub ) {
				dedisp_size chan_idx = (chan_word + chan_sub)*chan_stride
					+ batch_idx*batch_chan_stride;
				
				// Look up the fractional delay
				dedisp_float frac_delay = c_delay_table[chan_idx];
				// Compute the integer delay
				dedisp_size delay = __float2uint_rn(dm * frac_delay);
				
				if( USE_TEXTURE_MEM ) { // Pre-Fermi path
					// Loop over samples per thread
					// Note: Unrolled to ensure the sum[] array is stored in regs
                    #pragma unroll
					for( dedisp_size s=0; s<SAMPS_PER_THREAD; ++s ) {
						// Grab the word containing the sample from texture mem
						dedisp_word sample = tex1Dfetch(t_in, offset+s + delay);
						
						// Extract the desired subword and accumulate
						sum[s] +=
							__umul24(c_killmask[chan_idx],
									 extract_subword<IN_NBITS>(sample,chan_sub));
					}
				}
				else { // Post-Fermi path
					// Note: Unrolled to ensure the sum[] array is stored in regs
                    #pragma unroll
					for( dedisp_size s=0; s<SAMPS_PER_THREAD; ++s ) {
						// Grab the word containing the sample from global mem
						dedisp_word sample = d_in[offset + s + delay];
						
						// Extract the desired subword and accumulate
						sum[s] +=
							c_killmask[chan_idx] *
							extract_subword<IN_NBITS>(sample, chan_sub);
					}
				}
			}
		}
		
		// Write sums to global mem
		// Note: This is ugly, but easy, and doesn't hurt performance
		switch( out_nbits ) {
			case 8:
                #pragma unroll
				for( dedisp_size s=0; s<SAMPS_PER_THREAD; ++s ) {
					if( samp_idx*SAMPS_PER_THREAD + s < nsamps )
						((unsigned char*)d_out)[samp_idx*SAMPS_PER_THREAD + s +
												dm_idx * out_stride +
												batch_idx * batch_out_stride] =
							scale_output<IN_NBITS,unsigned char>(sum[s], nchans);
				}
				break;
			case 16:
                #pragma unroll
				for( dedisp_size s=0; s<SAMPS_PER_THREAD; ++s ) {
					if( samp_idx*SAMPS_PER_THREAD + s < nsamps )
						((unsigned short*)d_out)[samp_idx*SAMPS_PER_THREAD + s +
												 dm_idx * out_stride +
												 batch_idx * batch_out_stride] =
							scale_output<IN_NBITS,unsigned short>(sum[s], nchans);
				}
				break;
			case 32:
                #pragma unroll
				for( dedisp_size s=0; s<SAMPS_PER_THREAD; ++s ) {
					if( samp_idx*SAMPS_PER_THREAD + s < nsamps )
						((unsigned int*)d_out)[samp_idx*SAMPS_PER_THREAD + s +
											   dm_idx * out_stride +
											   batch_idx * batch_out_stride] =
							scale_output<IN_NBITS,unsigned int>(sum[s], nchans);
				}
				break;
			default:
				// Error
				break;
		}
		
	} // End of sample loop
	
	} // End of DM loop
}

bool check_use_texture_mem() {
	// Decides based on pre/post Fermi architecture
	int device_idx;
	cudaGetDevice(&device_idx);
	cudaDeviceProp device_props;
	cudaGetDeviceProperties(&device_props, device_idx);
	bool use_texture_mem = device_props.major < 2;
	return use_texture_mem;
}

bool dedisperse(const dedisp_word*  d_in,
                dedisp_size         in_stride,
                dedisp_size         nsamps,
                dedisp_size         in_nbits,
                dedisp_size         nchans,
                dedisp_size         chan_stride,
                const dedisp_float* d_dm_list,
                dedisp_size         dm_count,
                dedisp_size         dm_stride,
                dedisp_byte*        d_out,
                dedisp_size         out_stride,
                dedisp_size         out_nbits,
                dedisp_size         batch_size,
                dedisp_size         batch_in_stride,
                dedisp_size         batch_dm_stride,
                dedisp_size         batch_chan_stride,
                dedisp_size         batch_out_stride) {
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
	// --------------------------------------
	// Determine whether we should use texture memory
	bool use_texture_mem = check_use_texture_mem();
	if( use_texture_mem ) {
		dedisp_size chans_per_word = sizeof(dedisp_word)*BITS_PER_BYTE / in_nbits;
		dedisp_size nchan_words    = nchans / chans_per_word;
		dedisp_size input_words    = in_stride * nchan_words;
		
		// Check the texture size limit
		if( input_words > MAX_CUDA_1D_TEXTURE_SIZE ) {
			return false;
		}
		// Bind the texture memory
		cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<dedisp_word>();
		cudaBindTexture(0, t_in, d_in, channel_desc,
						input_words * sizeof(dedisp_word));
#ifdef DEDISP_DEBUG
		cudaError_t cuda_error = cudaGetLastError();
		if( cuda_error != cudaSuccess ) {
			return false;
		}
#endif // DEDISP_DEBUG
	}
	// --------------------------------------
	
	// Define thread decomposition
	// Note: Block dimensions x and y represent time samples and DMs respectively
	dim3 block(BLOCK_DIM_X,
			   BLOCK_DIM_Y);
	// Note: Grid dimension x represents time samples and DMs flattened
	//         together. Dimension y is used for batch jobs.
	
	// Divide and round up
	dedisp_size nsamp_blocks = (nsamps - 1)   
		/ ((dedisp_size)DEDISP_SAMPS_PER_THREAD*block.x) + 1;
	dedisp_size ndm_blocks   = (dm_count - 1) / (dedisp_size)block.y + 1;
	
	// Constrain the grid size to the maximum allowed
	// TODO: Consider cropping the batch size dimension instead and looping over it
	//         inside the kernel
	ndm_blocks = min((unsigned int)ndm_blocks,
					 (unsigned int)(MAX_CUDA_GRID_SIZE_Y/batch_size));
	
	// Note: We combine the DM and batch dimensions into one
	dim3 grid(nsamp_blocks,
			  ndm_blocks * batch_size);
	
	// Divide and round up
	dedisp_size nsamps_reduced = (nsamps - 1) / DEDISP_SAMPS_PER_THREAD + 1;
	
	cudaStream_t stream = 0;
	
	// Execute the kernel
#define DEDISP_CALL_KERNEL(NBITS, USE_TEXTURE_MEM)						\
	dedisperse_kernel<NBITS,DEDISP_SAMPS_PER_THREAD,BLOCK_DIM_X,        \
		              BLOCK_DIM_Y,USE_TEXTURE_MEM>                      \
		<<<grid, block, 0, stream>>>(d_in,								\
									 nsamps,							\
									 nsamps_reduced,					\
									 nsamp_blocks,						\
									 in_stride,							\
									 dm_count,							\
									 dm_stride,							\
									 ndm_blocks,						\
									 nchans,							\
									 chan_stride,						\
									 d_out,								\
									 out_nbits,							\
									 out_stride,						\
									 d_dm_list,							\
									 batch_in_stride,					\
									 batch_dm_stride,					\
									 batch_chan_stride,					\
									 batch_out_stride)
	// Note: Here we dispatch dynamically on nbits for supported values
	if( use_texture_mem ) {
		switch( in_nbits ) {
			case 1:  DEDISP_CALL_KERNEL(1,true);  break;
			case 2:  DEDISP_CALL_KERNEL(2,true);  break;
			case 4:  DEDISP_CALL_KERNEL(4,true);  break;
			case 8:  DEDISP_CALL_KERNEL(8,true);  break;
			case 16: DEDISP_CALL_KERNEL(16,true); break;
			case 32: DEDISP_CALL_KERNEL(32,true); break;
			default: /* should never be reached */ break;
		}
	}
	else {
		switch( in_nbits ) {
			case 1:  DEDISP_CALL_KERNEL(1,false);  break;
			case 2:  DEDISP_CALL_KERNEL(2,false);  break;
			case 4:  DEDISP_CALL_KERNEL(4,false);  break;
			case 8:  DEDISP_CALL_KERNEL(8,false);  break;
			case 16: DEDISP_CALL_KERNEL(16,false); break;
			case 32: DEDISP_CALL_KERNEL(32,false); break;
			default: /* should never be reached */ break;
		}
	}
#undef DEDISP_CALL_KERNEL
		
	// Check for kernel errors
#ifdef DEDISP_DEBUG
	//cudaStreamSynchronize(stream);
	cudaThreadSynchronize();
	cudaError_t cuda_error = cudaGetLastError();
	if( cuda_error != cudaSuccess ) {
		return false;
	}
#endif // DEDISP_DEBUG
	
	return true;
}