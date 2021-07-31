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
  This file contains function dedisperse_kernel
*/

// macros need to be defined when compile:
//     DEDISP_WORD_TYPE, DEDISP_SIZE_TYPE, DEDISP_FLOAT_TYPE, DEDISP_BYTE_TYPE, DEDISP_BOOL_TYPE
//     int IN_NBITS, int SAMPS_PER_THREAD, int BLOCK_DIM_X, int BLOCK_DIM_Y

// #define TEST_DEFINES
// test use only
#ifdef TEST_DEFINES 
#define DEDISP_WORD_TYPE uint
#define DEDISP_SIZE_TYPE ulong
#define DEDISP_FLOAT_TYPE float
#define DEDISP_BYTE_TYPE uchar
#define DEDISP_BOOL_TYPE int
#define IN_NBITS 16
// #define IN_NBITS 32
#define SAMPS_PER_THREAD 2
#define DEDISP_BLOCK_SAMPS 8
#define BLOCK_DIM_X DEDISP_BLOCK_SAMPS
#define DEDISP_BLOCK_SIZE 256
#define BLOCK_DIM_Y (DEDISP_BLOCK_SIZE / DEDISP_BLOCK_SAMPS)
#endif

typedef DEDISP_WORD_TYPE dedisp_word;
typedef DEDISP_SIZE_TYPE dedisp_size;
typedef DEDISP_FLOAT_TYPE dedisp_float;
typedef DEDISP_BYTE_TYPE dedisp_byte;
typedef DEDISP_BOOL_TYPE dedisp_bool;

// Summation type metafunction
/*
template<int IN_NBITS> struct SumType { typedef dedisp_word type; };
*/
// Note: For 32-bit input, we must accumulate using a larger data type
/*
template<> struct SumType<32> { typedef unsigned long long type; };
*/
#if IN_NBITS >= 32
typedef dedisp_word SumType;
#else
typedef ulong SumType;
#endif

/*
template<int NBITS, typename T=unsigned int>
struct max_value {
    static const T value = (((unsigned)1<<(NBITS-1))-1)*2+1;
};
*/
inline ulong max_value(int NBITS) {
    return (((unsigned)1<<(NBITS-1))-1)*2+1;
}

#define DEFINE_SCALE_OUTPUT(T) \
inline T scale_output_##T (SumType sum, dedisp_size nchans) { \
    enum { BITS_PER_BYTE = 8 }; \
    float in_range  = max_value(IN_NBITS); \
    /* Note: We use floats when out_nbits == 32, and scale to a range of [0:1] */ \
    float out_range = (sizeof(T)==4) ? 1.f \
                                     : max_value(sizeof(T)*BITS_PER_BYTE); \
 \
    /* Note: This emulates what dedisperse_all does for 2-bit HTRU data --> 8-bit \
             (and will adapt linearly to changes in in/out_nbits or nchans) */ \
    float factor = (3.f * 1024.f) / 255.f / 16.f; \
    float scaled = (float)sum * out_range / (in_range * nchans) * factor; \
    /* Clip to range when necessary */ \
    scaled = (sizeof(T) == 4) ? scaled : min(max(scaled, 0.f), out_range); \
    return (T)scaled; \
}
DEFINE_SCALE_OUTPUT(uchar)
DEFINE_SCALE_OUTPUT(ushort)
DEFINE_SCALE_OUTPUT(float)

// T = dedisp_word only
#define NBITS IN_NBITS
inline dedisp_word extract_subword(dedisp_word value, int idx) {
    dedisp_word MASK = (dedisp_word)max_value(NBITS);
    return (value >> (idx*NBITS)) & MASK;
}
#undef NBITS

/*
inline void set_out_val(__global dedisp_byte* d_out, dedisp_size idx,
                 SumType sum, dedisp_size nchans) {
    ((T*)d_out)[idx] = scale_output(sum, nchans);
}
*/

// Note: This assumes consecutive input words are consecutive times,
//         but that consecutive subwords are consecutive channels.
//       E.g., Words bracketed: (t0c0,t0c1,t0c2,t0c3), (t1c0,t1c1,t1c2,t1c3),...
// Note: out_stride should be in units of samples
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
            __constant dedisp_bool*        c_killmask)
{
    uint3 blockIdx = (uint3) ((uint) get_group_id(0), (uint) get_group_id(1), (uint) get_group_id(2));
    uint3 gridDim = (uint3) ((uint) get_num_groups(0), (uint) get_num_groups(1), (uint) get_num_groups(2));
    uint3 threadIdx = (uint3) ((uint) get_local_id(0), (uint) get_local_id(1), (uint) get_local_id(2));

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
    dedisp_float dm = d_dm_list[d_dm_list_offset + dm_idx*dm_stride + batch_idx*batch_dm_stride];
    
    // Loop over samples
    for( ; samp_idx < nsamps_reduced; samp_idx += nsamp_threads ) {
        // typedef typename SumType<IN_NBITS>::type sum_type;
        typedef SumType sum_type;
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
                dedisp_size delay = rint(dm * frac_delay);
                    // Note: Unrolled to ensure the sum[] array is stored in regs
                    #pragma unroll
                    for( dedisp_size s=0; s<SAMPS_PER_THREAD; ++s ) {
                        // Grab the word containing the sample from global mem
                        dedisp_word sample = d_in[d_in_offset + offset + s + delay];
                        
                        // Extract the desired subword and accumulate
                        sum[s] +=
                            c_killmask[chan_idx] *
                            extract_subword(sample, chan_sub);
                    }
//                }
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
                    if( samp_idx*SAMPS_PER_THREAD + s < nsamps ) {
                        /*
                        set_out_val<unsigned char, IN_NBITS>(d_out, out_idx + s,
                                                             sum[s], nchans);
                        */
                        
                        ((unsigned char*)(d_out[d_out_offset]))[out_idx + s] = scale_output_uchar(sum[s], nchans);
                    }
                }
                break;
            case 16:
                #pragma unroll
                for( dedisp_size s=0; s<SAMPS_PER_THREAD; ++s ) {
                    if( samp_idx*SAMPS_PER_THREAD + s < nsamps ) {
                        /*
                        set_out_val<unsigned short, IN_NBITS>(d_out, out_idx + s,
                                                              sum[s], nchans);
                        */
                        ((unsigned short*)(d_out[d_out_offset]))[out_idx + s] = scale_output_ushort(sum[s], nchans);
                    }
                }
                break;
            case 32:
                #pragma unroll
                for( dedisp_size s=0; s<SAMPS_PER_THREAD; ++s ) {
                    if( samp_idx*SAMPS_PER_THREAD + s < nsamps ) {
                        /*
                        set_out_val<float, IN_NBITS>(d_out, out_idx + s,
                                                     sum[s], nchans);
                        */
                        ((float*)(d_out[d_out_offset]))[out_idx + s] = scale_output_float(sum[s], nchans);
                    }
                }
                break;
            default:
                // Error
                break;
        }
        
    } // End of sample loop
    
    } // End of DM loop
}
