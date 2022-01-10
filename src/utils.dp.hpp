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
  This file contains functions that help migrating to OpenCL
*/

#pragma once

#include "dedisp.h"
#include <boost/compute.hpp>
#include <iostream>
#include <stdexcept>
#include <mutex>

typedef cl_uint gpu_size_t;

extern boost::compute::program_cache dedisp_program_cache;

namespace dedisp {

template<typename T>
inline std::string get_cl_typename() {
    auto error = []{
        throw std::runtime_error(std::string("Cannot convert type ") + std::string(typeid(T).name()));
    };
    std::string ret = "";
    if (std::is_floating_point<T>::value){
        switch (sizeof(T)) {
            case sizeof(cl_float):
                ret += "float";
                break;
            case sizeof(cl_double):
                ret += "double";
                break;
            default:
                error();
        }
    } else {
        if (std::is_unsigned<T>::value) {
            ret += "u";
        }
        switch (sizeof(T)) {
            case sizeof(cl_char):
                ret += "char";
                break;
            case sizeof(cl_short):
                ret += "short";
                break;
            case sizeof(cl_int):
                ret += "int";
                break;
            case sizeof(cl_long):
                ret += "long";
                break;
            default:
                error();
        }
    }
    return ret;
}

const std::string type_define_arguments = 
    std::string(" -DDEDISP_WORD_TYPE=") + dedisp::get_cl_typename<dedisp_word>() + " " +
    std::string(" -DDEDISP_SIZE_TYPE=") + dedisp::get_cl_typename<dedisp_size>() + " " +
    std::string(" -DDEDISP_FLOAT_TYPE=") + dedisp::get_cl_typename<dedisp_float>() + " " +
    std::string(" -DDEDISP_BYTE_TYPE=") + dedisp::get_cl_typename<dedisp_byte>() + " " +
    std::string(" -DDEDISP_BOOL_TYPE=") + dedisp::get_cl_typename<dedisp_bool>() + " ";

// reference: https://github.com/ROCm-Developer-Tools/ROCclr/blob/b8e8dc020ae79efbe703f9ca6c5e842b22e35850/device/gpu/gpukernel.cpp#L1069
inline size_t restrict_local_work_size(size_t global_size, size_t max_local_size) {
    size_t t = max_local_size;
    while (t--) {
        if (global_size % t == 0) {
            return t;
        }
    }
    return 1;
}

} // namespace dedisp
