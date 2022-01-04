/*
 *  Copyright 2022 fxzjshm
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

#pragma once

#include <CL/sycl.hpp>

// reference: dpct/device.hpp and boost/compute/system.hpp

namespace sycl = cl::sycl;

class device_ext {
public:
    device_ext(size_t device_id_) : device_id(device_id_) {
        std::vector<sycl::device> devices = sycl::device::get_devices();
        if (device_id >= devices.size()) {
            throw sycl::exception("device_ext: device_id " + std::to_string(device_id) + " out of bound" + std::to_string(devices.size()));
        }
        device = devices[device_id];
        queue = std::move(sycl::queue(device));
    }

    operator sycl::device () {
        return device;
    }

    operator sycl::device& () {
        return device;
    }

    inline size_t get_device_id() {
        return device_id;
    }

    inline sycl::queue& default_queue() {
        return queue;
    }

private:
    sycl::device device;
    size_t device_id;
    sycl::queue queue;
};

class dev_mgr {
public:
    static dev_mgr& instance() {
        static dev_mgr d_m;
        return d_m;
    }

    void select_device(size_t device_id) {
        device = std::move(device_ext(device_id));
    }

    inline sycl::device& current_device() {
        return device;
    }

    inline size_t current_device_id() {
        return device.get_device_id();
    }

    dev_mgr(const dev_mgr &) = delete;
    dev_mgr &operator=(const dev_mgr &) = delete;
    dev_mgr(dev_mgr &&) = delete;
    dev_mgr &operator=(dev_mgr &&) = delete;

private:
    device_ext device;

    dev_mgr() : device(0) {
        
    }
};