# The MIT License (MIT)

# Copyright (c) 2014-2022 David Medina and Tim Warburton

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

message(CHECK_START "Looking for DPCPP")
unset(missingDpcppComponents)

find_path(
  SYCL_INCLUDE_DIRS1
  NAMES
    sycl/sycl.hpp
  PATHS
    /opt/intel/oneapi/compiler/latest/linux
    ENV SYCL_ROOT
    ${SYCL_ROOT}
  PATH_SUFFIXES
    include
    include/sycl
)

find_path(
  SYCL_INCLUDE_DIRS2
  NAMES
    CL/sycl.hpp
  PATHS
    /opt/intel/oneapi/compiler/latest/linux
    ENV SYCL_ROOT
    ${SYCL_ROOT}
  PATH_SUFFIXES
    include/sycl
)
if(SYCL_INCLUDE_DIRS1 AND SYCL_INCLUDE_DIRS2)
  set(SYCL_INCLUDE_DIRS ${SYCL_INCLUDE_DIRS1} ${SYCL_INCLUDE_DIRS2})
endif()

find_library(
  SYCL_LIBRARIES
  NAMES
    sycl libsycl
  PATHS
    /opt/intel/oneapi/compiler/latest/linux
    ENV SYCL_ROOT
    ${SYCL_ROOT}
  PATH_SUFFIXES
    lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    DPCPP
    REQUIRED_VARS
    SYCL_INCLUDE_DIRS
    SYCL_LIBRARIES
    )

if(DPCPP_FOUND AND NOT TARGET nomp::SYCL)
  add_library(nomp::SYCL INTERFACE IMPORTED)
  set_target_properties(nomp::SYCL PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${SYCL_INCLUDE_DIRS}"
    INTERFACE_LINK_LIBRARIES "${SYCL_LIBRARIES}"
  )
endif()
