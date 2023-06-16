cmake_minimum_required(VERSION 3.22)
project(libnomp VERSION 0.0.1 DESCRIPTION "nomp runtime library" LANGUAGES C CXX)

# Treat compiler warnings as errors
set(CMAKE_COMPILE_WARNING_AS_ERROR ON)
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")

option(ENABLE_OPENCL "Build OpenCL backend" OFF)
option(ENABLE_CUDA "Build Cuda backend" OFF)
option(ENABLE_ISPC "Build ISPC backend" OFF)
option(ENABLE_HIP "Build HIP backend" OFF)
option(ENABLE_SYCL "Build SYCL backend" OFF)

# https://gitlab.kitware.com/cmake/community/-/wikis/doc/cmake/RPATH-handling
# Use, i.e. don't skip the full RPATH for the build tree
set(CMAKE_SKIP_BUILD_RPATH FALSE)
# When building, don't use the install RPATH
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
# Set the library location
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
# Add the automatically determined parts of the RPATH which point to directories
# outside the build tree to the install RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set(SOURCES src/nomp.c src/loopy.c src/aux.c src/log.c src/jit.c src/reduction.c)
if (ENABLE_OPENCL)
  find_package(OpenCL REQUIRED)
  list(APPEND SOURCES src/opencl.c)
endif()
if (ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  list(APPEND SOURCES src/cuda.c)
endif()
if (ENABLE_HIP)
  find_package(HIP)
  if(HIP_FOUND)
    list(APPEND SOURCES src/hip.c)
  else()
    message(FATAL_ERROR "ENABLE_HIP is ON but unable to find HIP runtime.")
  endif()
endif()
if (ENABLE_SYCL)
  find_package(DPCPP)
  if(DPCPP_FOUND)
    list(APPEND SOURCES src/sycl.cpp)
  else()
    message(FATAL_ERROR "ENABLE_SYCL is ON but unable to find SYCL runtime.")
  endif()
endif()
if (ENABLE_ISPC)
  find_package(OpenMP REQUIRED)
  find_package(ispcrt REQUIRED)
  list(APPEND SOURCES src/ispc.c)
endif()
add_library(nomp SHARED ${SOURCES})

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
target_link_libraries(nomp PRIVATE Python3::Python)
find_package(OpenSSL REQUIRED)
target_link_libraries(nomp PRIVATE OpenSSL::SSL)
find_package(SymEngine REQUIRED)
target_link_libraries(nomp PRIVATE nomp::SymEngine)
if (ENABLE_OPENCL)
  target_link_libraries(nomp PRIVATE OpenCL::OpenCL)
  target_compile_definitions(nomp PRIVATE OPENCL_ENABLED)
endif()
if (ENABLE_CUDA)
  target_link_libraries(nomp PRIVATE CUDA::cudart CUDA::nvrtc)
  target_compile_definitions(nomp PRIVATE CUDA_ENABLED)
endif()
if (ENABLE_HIP)
  target_link_libraries(nomp PRIVATE nomp::HIP)
  target_compile_definitions(nomp PRIVATE HIP_ENABLED)
endif()
if (ENABLE_SYCL)
  target_link_libraries(nomp PRIVATE nomp::SYCL)
  target_compile_definitions(nomp PRIVATE SYCL_ENABLED)
endif()
if (ENABLE_ISPC)
  target_link_libraries(nomp PRIVATE OpenMP::OpenMP_C ispcrt::ispcrt)
  target_compile_definitions(nomp PRIVATE ISPC_ENABLED)
endif()

set_target_properties(nomp PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION 1
  PUBLIC_HEADER src/nomp.h)
target_include_directories(nomp PRIVATE src)

option(ENABLE_DOCS "Enable Documentation" OFF)
if (ENABLE_DOCS)
  add_subdirectory ("docs")
endif()

install(TARGETS nomp LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib PUBLIC_HEADER
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/python DESTINATION ${CMAKE_INSTALL_PREFIX})

configure_file(nomp.pc.in nomp.pc @ONLY)
install(FILES ${CMAKE_BINARY_DIR}/nomp.pc DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/pkgconfig)

file(GLOB TESTS tests/nomp-*.c)
foreach(test ${TESTS})
  string(REPLACE "${CMAKE_SOURCE_DIR}/tests/" "" temp ${test})
  string(REPLACE ".c" "" test_exe ${temp})
  add_executable(${test_exe} ${test})
  target_link_libraries(${test_exe} nomp)
  target_include_directories(${test_exe} PUBLIC src tests)
  install (TARGETS ${test_exe} RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/tests)
endforeach(test ${TESTS})
install(DIRECTORY tests/ DESTINATION ${CMAKE_INSTALL_PREFIX}/tests
  FILES_MATCHING PATTERN "*.py")
install(DIRECTORY scripts/ DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
  FILE_PERMISSIONS OWNER_READ OWNER_EXECUTE OWNER_WRITE GROUP_READ GROUP_EXECUTE
  WORLD_READ WORLD_EXECUTE)
