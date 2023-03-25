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

if(DPCPP_FOUND AND NOT TARGET DPCPP::SYCL)
  add_library(DPCPP::SYCL INTERFACE IMPORTED)
  set_target_properties(DPCPP::SYCL PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${SYCL_INCLUDE_DIRS}"
    INTERFACE_LINK_LIBRARIES "${SYCL_LIBRARIES}"
  )
endif()
