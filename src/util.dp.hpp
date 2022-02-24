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

#include <boost/iterator/counting_iterator.hpp>
#include <boost/iterator/permutation_iterator.hpp>

#if defined(SYCL_DEVICE_COPYABLE) && SYCL_DEVICE_COPYABLE
// patch for foreign iterators
template <typename T>
struct sycl::is_device_copyable<boost::iterators::counting_iterator<T>> : std::true_type {};
template <class ElementIterator, class IndexIterator>
struct sycl::is_device_copyable<boost::iterators::permutation_iterator<ElementIterator, IndexIterator>, std::enable_if_t<!std::is_trivially_copyable<boost::iterators::permutation_iterator<ElementIterator, IndexIterator>>::value>> : std::true_type {};
#endif