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
  This file contains functions scrunch_x2_kernel
*/

// macros need to be defined when compile:
//     DEDISP_WORD_TYPE, DEDISP_SIZE_TYPE

typedef DEDISP_WORD_TYPE dedisp_word;
typedef DEDISP_SIZE_TYPE dedisp_size;
typedef dedisp_word WordType;

__kernel void scrunch_x2_kernel(__global WordType* in, dedisp_size in_offset, __global dedisp_word* outs, dedisp_size out_offset, int nbits, unsigned int in_nsamps) {
    unsigned int out_nsamps = in_nsamps / 2;
    unsigned int mask = (1<<nbits)-1;

    unsigned int out_i = get_global_id(0);

    unsigned int c     = out_i / out_nsamps;
    unsigned int out_t = out_i % out_nsamps;
    unsigned int in_t0 = out_t * 2;
    unsigned int in_t1 = out_t * 2 + 1;
    unsigned int in_i0 = c * in_nsamps + in_t0;
    unsigned int in_i1 = c * in_nsamps + in_t1;
    
    dedisp_word in0 = in[in_offset + in_i0];
    dedisp_word in1 = in[in_offset + in_i1];
    dedisp_word out = 0;
    for( int k=0; k<sizeof(WordType)*8; k+=nbits ) {
        dedisp_word s0 = (in0 >> k) & mask;
        dedisp_word s1 = (in1 >> k) & mask;
        dedisp_word avg = ((unsigned long long)s0 + s1) / 2;
        out |= avg << k;
    }
    outs[out_offset + out_i] = out;
}
