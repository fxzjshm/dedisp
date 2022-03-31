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

//#include "device_manager.dp.hpp"
#include "dpct/device.hpp"

//#include "device_vector.dp.hpp"
#include "dpct/dpl_extras/vector.h"

#include "dpct/dpl_extras/iterators.h"

#include <sycl/execution_policy>

extern ::sycl::sycl_execution_policy<> execution_policy;

template <typename T, sycl::usm::alloc AllocKind = sycl::usm::alloc::device,
          size_t align = sizeof(T)>
class device_allocator {
public:
  device_allocator(cl::sycl::queue &queue_) : queue(queue_){};

  T *allocate(std::size_t num_elements) {
    T *ptr = sycl::aligned_alloc_device<T>(align, num_elements, queue);
    if (!ptr)
      throw std::runtime_error("device_allocator: Allocation failed");
    return ptr;
  }

  void deallocate(T *ptr, std::size_t size) {
    if (ptr)
      sycl::free(ptr, queue);
  }

private:
  cl::sycl::queue queue;
};

#ifndef DPCT_USM_LEVEL_NONE
template <typename T,
          typename Allocator = device_allocator<T> /*sycl::usm_allocator<T, sycl::usm::alloc::shared>*/ >
#else
template <typename T, typename Allocator = cl::sycl::buffer_allocator>
#endif
class device_vector_wrapper : public dpct::device_vector<T, Allocator> {
public:
    using size_type = std::size_t;
    using dpct::device_vector<T, Allocator>::device_vector;

    template <typename OtherAllocator>
    dpct::device_vector<T, Allocator> &operator=(const std::vector<T, OtherAllocator> &v) {
        return dpct::device_vector<T, Allocator>::operator=(v);
    }

    void resize(size_type new_size, const T &x = T()) {
        size_type old_size = dpct::device_vector<T, Allocator>::size();
        dpct::device_vector<T, Allocator>::resize(new_size, x);
        // wait here as operations above may be async, otherwise iterators may be invalid if memory is reallocated
        dpct::get_default_queue().wait();
        if (old_size < new_size) {
            ::sycl::impl::fill(execution_policy,
                dpct::device_vector<T, Allocator>::begin() + old_size, dpct::device_vector<T, Allocator>::begin() + new_size, x
            );
            execution_policy.get_queue().wait();
        }
    }
};

#include <boost/iterator/counting_iterator.hpp>
#include <boost/iterator/permutation_iterator.hpp>

#if defined(SYCL_DEVICE_COPYABLE) && SYCL_DEVICE_COPYABLE
// patch for foreign iterators
template <typename T>
struct sycl::is_device_copyable<boost::iterators::counting_iterator<T>> : std::true_type {};
template <class ElementIterator, class IndexIterator>
struct sycl::is_device_copyable<boost::iterators::permutation_iterator<ElementIterator, IndexIterator>, std::enable_if_t<!std::is_trivially_copyable<boost::iterators::permutation_iterator<ElementIterator, IndexIterator>>::value>> : std::true_type {};
#endif