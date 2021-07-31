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
  This file contains functions unpack_kernel
*/

// macros need to be defined when compile:
//     DEDISP_WORD_TYPE

typedef DEDISP_WORD_TYPE dedisp_word;
typedef dedisp_word WordType;

__kernel void unpack_kernel(__global WordType* in, __global dedisp_word* out, int nsamps, int in_nbits, int out_nbits) {
    uint i = get_global_id(0);
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
    out[i] = result;
}