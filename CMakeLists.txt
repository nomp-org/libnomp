cmake_minimum_required(VERSION 3.22)
project(libnomp VERSION 0.0.1 DESCRIPTION "nomp runtime library" LANGUAGES C)

option(ENABLE_OPENCL "Build OpenCL backend" ON)
option(ENABLE_CUDA "Build Cuda backend" OFF)

# https://gitlab.kitware.com/cmake/community/-/wikis/doc/cmake/RPATH-handling
# Use, i.e. don't skip the full RPATH for the build tree
set(CMAKE_SKIP_BUILD_RPATH FALSE)
# When building, don't use the install RPATH
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
# Add the automatically determined parts of the RPATH which point to directories
# outside the build tree to the install RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set(SOURCES src/nomp.c src/loopy.c src/auxiliary.c)
if (ENABLE_OPENCL)
  list(APPEND SOURCES src/opencl.c)
  add_compile_definitions(OPENCL_ENABLED)
endif()
if (ENABLE_CUDA)
  list(APPEND SOURCES src/cuda.c)
  add_compile_definitions(CUDA_ENABLED)
endif()
add_library(nomp SHARED ${SOURCES})

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
target_link_libraries(nomp PRIVATE Python3::Python)
if (ENABLE_OPENCL)
  find_package(OpenCL REQUIRED)
  target_link_libraries(nomp PRIVATE OpenCL::OpenCL)
endif()
if (ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(nomp PRIVATE CUDA::cudart CUDA::nvrtc)
endif()

set_target_properties(nomp PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION 1
  PUBLIC_HEADER src/nomp.h)
target_include_directories(nomp PRIVATE src)

file(GLOB TESTS tests/nomp-api-0*.c tests/nomp-api-1*.c tests/nomp-api-20*.c tests/nomp-api-21*.c)
foreach(test ${TESTS})
  string(REPLACE "${CMAKE_SOURCE_DIR}/tests/" "" temp ${test})
  string(REPLACE ".c" "" test_exe ${temp})
  add_executable(${test_exe} ${test})
  target_link_libraries(${test_exe} nomp)
  target_include_directories(${test_exe} PUBLIC src tests)
endforeach(test ${TESTS})

option(ENABLE_DOCS "Enable Documentation" OFF)
if (ENABLE_DOCS)
  set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
  add_subdirectory ("docs")
endif()

install(TARGETS nomp LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib PUBLIC_HEADER
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/python DESTINATION ${CMAKE_INSTALL_PREFIX})

configure_file(nomp.pc.in nomp.pc @ONLY)
install(FILES ${CMAKE_BINARY_DIR}/nomp.pc DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/pkgconfig)
