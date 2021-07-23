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

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <vector>
#include <cmath>

#include <algorithm>
 // For generate_dm_list

 // For scrunch_x2

// Kernel tuning parameters
#define DEDISP_BLOCK_SIZE       256
#define DEDISP_BLOCK_SAMPS      8
#define DEDISP_SAMPS_PER_THREAD 2 // 4 is better for Fermi?

dpct::constant_memory<dedisp_float, 1> c_delay_table(DEDISP_MAX_NCHANS);
dpct::constant_memory<dedisp_bool, 1> c_killmask(DEDISP_MAX_NCHANS);

// Texture reference for input data
/*
DPCT1059:10: SYCL only supports 4-channel image format. Adjust the code.
*/
dpct::image_wrapper<unsigned int, 1> t_in;

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

template<int IN_NBITS, typename T, typename SumType>
inline 
T scale_output(SumType sum, dedisp_size nchans) {
	enum { BITS_PER_BYTE = 8 };
	// This emulates dedisperse_all, but is specific to 8-bit output
	// Note: This also breaks the sub-band algorithm
	//return (dedisp_word)(((unsigned int)sum >> 4) - 128 - 128);
	// HACK
	//return (T)(sum / 16 - 128 - 128);
	
	// This uses the full range of the output bits
	//return (T)((double)sum / ((double)nchans * max_value<IN_NBITS>::value)
	//		   * max_value<sizeof(T)*BITS_PER_BYTE>::value );
	/*
	// This assumes the input data have mean=range/2 and then scales by
	//   assuming the SNR goes like sqrt(nchans).
	double in_range  = max_value<IN_NBITS>::value;
	double mean      = 0.5 *      (double)nchans  * in_range;
	double max_val   = 0.5 * sqrt((double)nchans) * in_range;
	
	// TODO: There are problems with the output scaling when in_nbits is small
	//         (e.g., in_nbits < 8). Not sure what to do about it at this stage.
	
	// TESTING This fixes 1-bit
	// TODO: See test_quantised_rms.py for further exploration of this
	//double max       = 0.5 * sqrt((double)nchans) * in_range * 2*4.545454; // HACK
	// TESTING This fixes 2-bit
	//double max       = 0.5 * sqrt((double)nchans) * in_range * 0.8*4.545454; // HACK
	// TESTING This fixes 4-bit
	//double max       = 0.5 * sqrt((double)nchans) * in_range * 0.28*4.545454; // HACK
	double out_range = max_value<sizeof(T)*BITS_PER_BYTE>::value;
	double out_mean  = 0.5 * out_range;
	double out_max   = 0.5 * out_range;
	double scaled = floor((sum-mean)/max_val * out_max + out_mean + 0.5);
	*/
	float in_range  = max_value<IN_NBITS>::value;
	// Note: We use floats when out_nbits == 32, and scale to a range of [0:1]
	float out_range = (sizeof(T)==4) ? 1.f
	                                 : max_value<sizeof(T)*BITS_PER_BYTE>::value;
	//float scaled = (float)sum / in_range / sqrt((float)nchans) * out_range;
	//float scaled = (float)sum / (in_range * nchans) * out_range;
	//float scaled = sum * ((float)out_range / in_range / 85.f) / 16.f;
	
	// Note: This emulates what dedisperse_all does for 2-bit HTRU data --> 8-bit
	//         (and will adapt linearly to changes in in/out_nbits or nchans)
	float factor = (3.f * 1024.f) / 255.f / 16.f;
	float scaled = (float)sum * out_range / (in_range * nchans) * factor;
	// Clip to range when necessary
        scaled = (sizeof(T) == 4) ? scaled
                                  : sycl::min(sycl::max(scaled, 0.), out_range);
        return (T)scaled;
}

template<int NBITS, typename T>
inline 
T extract_subword(T value, int idx) {
	enum { MASK = max_value<NBITS,T>::value };
	return (value >> (idx*NBITS)) & MASK;
}

// Summation type metafunction
template<int IN_NBITS> struct SumType { typedef dedisp_word type; };
// Note: For 32-bit input, we must accumulate using a larger data type
template<> struct SumType<32> { typedef unsigned long long type; };

template<typename T, int IN_NBITS, typename SumType>
inline 
void set_out_val(dedisp_byte* d_out, dedisp_size idx,
                 SumType sum, dedisp_size nchans) {
	((T*)d_out)[idx] = scale_output<IN_NBITS,T>(sum, nchans);
}

// Note: This assumes consecutive input words are consecutive times,
//         but that consecutive subwords are consecutive channels.
//       E.g., Words bracketed: (t0c0,t0c1,t0c2,t0c3), (t1c0,t1c1,t1c2,t1c3),...
// Note: out_stride should be in units of samples
template<int IN_NBITS, int SAMPS_PER_THREAD,
		 int BLOCK_DIM_X, int BLOCK_DIM_Y,
		 bool USE_TEXTURE_MEM>

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
                       dedisp_size         batch_out_stride,
                       sycl::nd_item<3> item_ct1,
                       dedisp_float *c_delay_table,
                       dedisp_bool *c_killmask,
                       dpct::image_accessor_ext<unsigned int, 1> t_in)
{
	// Compute compile-time constants
	enum {
		BITS_PER_BYTE  = 8,
		CHANS_PER_WORD = sizeof(dedisp_word) * BITS_PER_BYTE / IN_NBITS
	};
	
	// Compute the thread decomposition
        dedisp_size samp_block = item_ct1.get_group(2);
        dedisp_size dm_block = item_ct1.get_group(1) % ndm_blocks;
        dedisp_size batch_block = item_ct1.get_group(1) / ndm_blocks;

        dedisp_size samp_idx = samp_block * BLOCK_DIM_X + item_ct1.get_local_id(2);
        dedisp_size dm_idx = dm_block * BLOCK_DIM_Y + item_ct1.get_local_id(1);
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
                                dedisp_size delay =
                                    sycl::vec<float, 1>{dm * frac_delay}
                                        .convert<unsigned int,
                                                 sycl::rounding_mode::rte>()[0];

                                if( USE_TEXTURE_MEM ) { // Pre-Fermi path
					// Loop over samples per thread
					// Note: Unrolled to ensure the sum[] array is stored in regs
                    #pragma unroll
					for( dedisp_size s=0; s<SAMPS_PER_THREAD; ++s ) {
						// Grab the word containing the sample from texture mem
                                                dedisp_word sample = t_in.read(offset + s + delay);

                                                // Extract the desired subword and accumulate
						sum[s] +=
							// TODO: Pre-Fermi cards are faster with 24-bit mul
							/*__umul24*/(c_killmask[chan_idx] *//,
									 extract_subword<IN_NBITS>(sample,chan_sub));
					}
				}
				else { // Fermi path
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
		dedisp_size out_idx = ( samp_idx*SAMPS_PER_THREAD +
		                        dm_idx * out_stride +
		                        batch_idx * batch_out_stride );
		switch( out_nbits ) {
			case 8:
                #pragma unroll
				for( dedisp_size s=0; s<SAMPS_PER_THREAD; ++s ) {
					if( samp_idx*SAMPS_PER_THREAD + s < nsamps )
						set_out_val<unsigned char, IN_NBITS>(d_out, out_idx + s,
						                                     sum[s], nchans);
				}
				break;
			case 16:
                #pragma unroll
				for( dedisp_size s=0; s<SAMPS_PER_THREAD; ++s ) {
					if( samp_idx*SAMPS_PER_THREAD + s < nsamps )
						set_out_val<unsigned short, IN_NBITS>(d_out, out_idx + s,
						                                      sum[s], nchans);
				}
				break;
			case 32:
                #pragma unroll
				for( dedisp_size s=0; s<SAMPS_PER_THREAD; ++s ) {
					if( samp_idx*SAMPS_PER_THREAD + s < nsamps )
						set_out_val<float, IN_NBITS>(d_out, out_idx + s,
						                             sum[s], nchans);
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
	// Decides based on GPU architecture
	int device_idx;
        device_idx = dpct::dev_mgr::instance().current_device_id();
        dpct::device_info device_props;
        dpct::dev_mgr::instance().get_device(device_idx).get_device_info(device_props);
        // Fermi runs worse with texture mem
        /*
        DPCT1005:11: The SYCL device version is different from CUDA Compute
        Compatibility. You may need to rewrite this code.
        */
        bool use_texture_mem = (device_props.get_major_version() < 2);
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
                /*
                DPCT1059:12: SYCL only supports 4-channel image format. Adjust
                the code.
                */
                dpct::image_channel channel_desc = dpct::image_channel::create<dedisp_word>();
                t_in.attach(d_in, input_words * sizeof(dedisp_word), channel_desc);
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
        sycl::range<3> block(1, BLOCK_DIM_Y, BLOCK_DIM_X);
        // Note: Grid dimension x represents time samples. Dimension y represents
	//         DMs and batch jobs flattened together.
	
	// Divide and round up
        dedisp_size nsamp_blocks =
            (nsamps - 1) / ((dedisp_size)DEDISP_SAMPS_PER_THREAD * block[2]) +
            1;
        dedisp_size ndm_blocks = (dm_count - 1) / (dedisp_size)block[1] + 1;

        // Constrain the grid size to the maximum allowed
	// TODO: Consider cropping the batch size dimension instead and looping over it
	//         inside the kernel
        ndm_blocks =
            std::min((unsigned int)ndm_blocks,
                     (unsigned int)(MAX_CUDA_GRID_SIZE_Y / batch_size));

        // Note: We combine the DM and batch dimensions into one
        sycl::range<3> grid(1, ndm_blocks * batch_size, nsamp_blocks);

        // Divide and round up
	dedisp_size nsamps_reduced = (nsamps - 1) / DEDISP_SAMPS_PER_THREAD + 1;

        sycl::queue *stream = 0;

        // Execute the kernel
/*
DPCT1049:13: The workgroup size passed to the SYCL kernel may exceed the limit.
To get the device limit, query info::device::max_work_group_size. Adjust the
workgroup size if needed.
*/
#define DEDISP_CALL_KERNEL(NBITS, USE_TEXTURE_MEM)                             \
        stream->submit(\
  [&](sycl::handler &cgh) {\
    // init global memory\
    c_delay_table.init(*stream);\
    c_killmask.init(*stream);\
\
    // pointers to device memory\
    auto c_delay_table_ptr_ct1 = c_delay_table.get_ptr();\
    auto c_killmask_ptr_ct1 = c_killmask.get_ptr();\
\
    // accessors to image objects\
    auto t_in_acc = t_in.get_access(cgh);\
\
    // sampler of image objects\
    auto t_in_smpl = t_in.get_sampler();\
\
    // helper variables defined\
    auto d_in_ct0 = d_in;\
    auto nsamps_ct1 = nsamps;\
    auto nsamps_reduced_ct2 = nsamps_reduced;\
    auto nsamp_blocks_ct3 = nsamp_blocks;\
    auto in_stride_ct4 = in_stride;\
    auto dm_count_ct5 = dm_count;\
    auto dm_stride_ct6 = dm_stride;\
    auto ndm_blocks_ct7 = ndm_blocks;\
    auto nchans_ct8 = nchans;\
    auto chan_stride_ct9 = chan_stride;\
    auto d_out_ct10 = d_out;\
    auto out_nbits_ct11 = out_nbits;\
    auto out_stride_ct12 = out_stride;\
    auto d_dm_list_ct13 = d_dm_list;\
    auto batch_in_stride_ct14 = batch_in_stride;\
    auto batch_dm_stride_ct15 = batch_dm_stride;\
    auto batch_chan_stride_ct16 = batch_chan_stride;\
    auto batch_out_stride_ct17 = batch_out_stride;\
\
    cgh.parallel_for(\
      sycl::nd_range<3>(grid * block, block), \
      [=](sycl::nd_item<3> item_ct1) {\
        dedisperse_kernel<NBITS, DEDISP_CALL_KERNEL, DEDISP_CALL, DEDISP_CALL, USE_TEXTURE_MEM>(d_in_ct0, nsamps_ct1, nsamps_reduced_ct2, nsamp_blocks_ct3, in_stride_ct4, dm_count_ct5, dm_stride_ct6, ndm_blocks_ct7, nchans_ct8, chan_stride_ct9, d_out_ct10, out_nbits_ct11, out_stride_ct12, d_dm_list_ct13, batch_in_stride_ct14, batch_dm_stride_ct15, batch_chan_stride_ct16, batch_out_stride_ct17, item_ct1, c_delay_table_ptr_ct1, c_killmask_ptr_ct1, dpct::image_accessor_ext<unsigned int, 1>(t_in_smpl, t_in_acc));\
      });\
  });
	// Note: Here we dispatch dynamically on nbits for supported values
        if( use_texture_mem ) {
		switch( in_nbits ) {
                        case 1: DEDISP_CALL_KERNEL(1, true) break;
                        case 2: DEDISP_CALL_KERNEL(2, true) break;
                        case 4: DEDISP_CALL_KERNEL(4, true) break;
                        case 8: DEDISP_CALL_KERNEL(8, true) break;
                        case 16: DEDISP_CALL_KERNEL(16, true) break;
                        case 32: DEDISP_CALL_KERNEL(32, true) break;
                        default: /* should never be reached */ break;
		}
	}
	else {
		switch( in_nbits ) {
                        case 1: DEDISP_CALL_KERNEL(1, false) break;
                        case 2: DEDISP_CALL_KERNEL(2, false) break;
                        case 4: DEDISP_CALL_KERNEL(4, false) break;
                        case 8: DEDISP_CALL_KERNEL(8, false) break;
                        case 16: DEDISP_CALL_KERNEL(16, false) break;
                        case 32: DEDISP_CALL_KERNEL(32, false) break;
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


template<typename WordType>
struct scrunch_x2_functor
        /*
DPCT1044:14: thrust::unary_function was removed because std::unary_function has
been deprecated in C++11. You may need to remove references to typedefs from
thrust::unary_function in the class definition.
*/
        {
        const WordType* in;
	int             nbits;
	WordType        mask;
	unsigned int    in_nsamps;
	unsigned int    out_nsamps;
	scrunch_x2_functor(const WordType* in_, int nbits_, unsigned int in_nsamps_)
		: in(in_), nbits(nbits_), mask((1<<nbits)-1),
		  in_nsamps(in_nsamps_), out_nsamps(in_nsamps_/2) {}
	inline 
	WordType operator()(unsigned int out_i) const {
		unsigned int c     = out_i / out_nsamps;
		unsigned int out_t = out_i % out_nsamps;
		unsigned int in_t0 = out_t * 2;
		unsigned int in_t1 = out_t * 2 + 1;
		unsigned int in_i0 = c * in_nsamps + in_t0;
		unsigned int in_i1 = c * in_nsamps + in_t1;
		
		dedisp_word in0 = in[in_i0];
		dedisp_word in1 = in[in_i1];
		dedisp_word out = 0;
		for( int k=0; k<sizeof(WordType)*8; k+=nbits ) {
			dedisp_word s0 = (in0 >> k) & mask;
			dedisp_word s1 = (in1 >> k) & mask;
			dedisp_word avg = ((unsigned long long)s0 + s1) / 2;
			out |= avg << k;
		}
		return out;
	}
};

// Reduces the time resolution by 2x
dedisp_error scrunch_x2(const dedisp_word* d_in,
                        dedisp_size nsamps,
                        dedisp_size nchan_words,
                        dedisp_size nbits,
                        dedisp_word* d_out)
{
        dpct::device_pointer<dedisp_word> d_out_begin(d_out);

        dedisp_size out_nsamps = nsamps / 2;
	dedisp_size out_count  = out_nsamps * nchan_words;
	
	using thrust::make_counting_iterator;

        std::transform(oneapi::dpl::execution::make_device_policy(
                           dpct::get_default_queue()),
                       dpct::make_counting_iterator<unsigned int>(0),
                       dpct::make_counting_iterator<unsigned int>(out_count),
                       d_out_begin,
                       scrunch_x2_functor<dedisp_word>(d_in, nbits, nsamps));

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

template<typename WordType>
struct unpack_functor
        /*
DPCT1044:15: thrust::unary_function was removed because std::unary_function has
been deprecated in C++11. You may need to remove references to typedefs from
thrust::unary_function in the class definition.
*/
        {
        const WordType* in;
	int             nsamps;
	int             in_nbits;
	int             out_nbits;
	unpack_functor(const WordType* in_, int nsamps_, int in_nbits_, int out_nbits_)
		: in(in_), nsamps(nsamps_), in_nbits(in_nbits_), out_nbits(out_nbits_) {}
	inline 
	WordType operator()(unsigned int i) const {
		int out_chans_per_word = sizeof(WordType)*8 / out_nbits;
		int in_chans_per_word = sizeof(WordType)*8 / in_nbits;
		//int expansion = out_nbits / in_nbits;
		int norm = ((1l<<out_nbits)-1) / ((1l<<in_nbits)-1);
		WordType in_mask  = (1<<in_nbits)-1;
		WordType out_mask = (1<<out_nbits)-1;
		
		/*
		  cw\k 0123 0123
		  0    0123|0123
		  1    4567|4567
		  
		  cw\k 0 1
		  0    0 1 | 0 1
		  1    2 3 | 2 3
		  2    4 5 | 4 5
		  3    6 7 | 6 7
		  
		  
		 */
		
		unsigned int t      = i % nsamps;
		// Find the channel word indices
		unsigned int out_cw = i / nsamps;
		//unsigned int in_cw  = out_cw / expansion;
		//unsigned int in_i   = in_cw * nsamps + t;
		//WordType word = in[in_i];
		
		WordType result = 0;
		for( int k=0; k<sizeof(WordType)*8; k+=out_nbits ) {
			
			int c = out_cw * out_chans_per_word + k/out_nbits;
			int in_cw = c / in_chans_per_word;
			int in_k  = c % in_chans_per_word * in_nbits;
			int in_i  = in_cw * nsamps + t;
			WordType word = in[in_i];
			
			WordType val = (word >> in_k) & in_mask;
			result |= ((val * norm) & out_mask) << k;
		}
		return result;
	}
};

dedisp_error unpack(const dedisp_word* d_transposed,
                    dedisp_size nsamps, dedisp_size nchan_words,
                    dedisp_word* d_unpacked,
                    dedisp_size in_nbits, dedisp_size out_nbits)
{
        dpct::device_pointer<dedisp_word> d_unpacked_begin(d_unpacked);

        dedisp_size expansion = out_nbits / in_nbits;
	dedisp_size in_count  = nsamps * nchan_words;
	dedisp_size out_count = in_count * expansion;
	
	using thrust::make_counting_iterator;

        std::transform(oneapi::dpl::execution::make_device_policy(
                           dpct::get_default_queue()),
                       dpct::make_counting_iterator<unsigned int>(0),
                       dpct::make_counting_iterator<unsigned int>(out_count),
                       d_unpacked_begin,
                       unpack_functor<dedisp_word>(d_transposed, nsamps,
                                                   in_nbits, out_nbits));

        return DEDISP_NO_ERROR;
}