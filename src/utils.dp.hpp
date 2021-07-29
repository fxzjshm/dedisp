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
#include <CL/opencl.hpp>
#include <iostream>
#include <stdexcept>
#include <mutex>

using namespace cl;

typedef cl_uint gpu_size_t;

namespace dedisp {

#define cl_check(error) { \
    std::cerr << "Error" << error << "at" << __FILE__ << ", " << "line" << __LINE__ << std::endl; \
    throw std::runtime_error(error); \
}

// acts like Intel's DPCT dpct/device.hpp
class device_ext {
public:
    device_ext(cl::Device d) {
        device = d;
        context = cl::Context(device);
        queue = cl::CommandQueue(context);
    }

    inline cl::CommandQueue default_queue() {
        return queue;
    }

    inline cl::Context default_context() {
        return context;
    }

    inline cl::Device unwrap() {
        return device;
    }

private:
    cl::Device device;
    cl::CommandQueue queue;
    cl::Context context;
};

// acts like Intel's DPCT dpct/device.hpp
class device_manager {
public:
    static device_manager& instance() {
        // TODO not thread safe before c++11
        static device_manager d_m;
        return d_m;
    }

    inline device_ext current_device() {
        return devices[id];
    }

    inline dedisp_size current_device_id() {
        return id;
    }

    inline cl::Context current_context() {
        return current_device().default_context();
    }

    inline cl::CommandQueue current_queue() {
        return current_device().default_queue();
    }

    inline cl_int select_device(dedisp_size _id) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (_id < devices.size()){
            id = _id;
            return CL_SUCCESS;
        } else {
            return CL_INVALID_ARG_VALUE;
        }
    }

private:
    mutable std::mutex m_mutex;
    std::vector<device_ext> devices;
    dedisp_size id;

    device_manager(){
        // for all platforms get all devices
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        for (auto &p : platforms) {
            std::vector<cl::Device> _devices;
            p.getDevices(CL_DEVICE_TYPE_ALL, &_devices);
            for(auto &d : _devices) {
                devices.push_back(device_ext(d));
            }
        }

        // select first one as the default
        select_device(0);
    }

    ~device_manager() = default;
    device_manager(const device_manager&) = delete;
	device_manager& operator=(const device_manager&) = delete;
    
};

// wrapper for cl::Buffer, emulates CUDA or DPCT's device_vector
template<typename T>
class device_vector {
public:
    device_vector() = default;

    device_vector(cl::Buffer buffer) {
        buf = buffer;
    }

    cl_uint size() {
        return buf.getInfo<CL_MEM_SIZE>();
    }

    void resize(dedisp_size n) {
        cl_int error;
        cl::CommandQueue& queue = device_manager::instance().current_queue();
        dedisp_size size = n * sizeof(T);
        cl::Buffer new_buf = cl::Buffer(device_manager::instance().current_context(), CL_MEM_READ_WRITE, size, nullptr, &error);
        cl_check(error);
        if (buf.get()) {
            error = queue.enqueueCopyBuffer(buf, new_buf, 0, 0, std::min(this.size(), size));
            cl_check(error);
            error = queue.finish();
            cl_check(error);
        }
        std::lock_guard<std::mutex> lock(m_mutex);
        buf = new_buf;
    }

    void operator=(std::vector<T>& vec) {
        queue.enqueueWriteBuffer(buf, /* blocking = */ CL_TRUE, 0, vec.size(), &vec[0]);
    }

    cl_int fill(T src) {
        return device_manager::instance().current_queue().enqueueFillBuffer(buf, src, 0, size());
    }

    cl::Buffer buffer() {
        return this->buf;
    }

    operator cl::Buffer() const {
        return this->buf;
    }

private:
    cl::Buffer buf;
    mutable std::mutex m_mutex;
};

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

} // namespace dedisp
