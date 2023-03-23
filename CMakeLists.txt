cmake_minimum_required(VERSION 3.22)
project(libnomp VERSION 0.0.1 DESCRIPTION "nomp runtime library" LANGUAGES C CXX)

# Treat compiler warnings as errors
set(CMAKE_COMPILE_WARNING_AS_ERROR ON)
set(CMAKE_C_STANDARD 99)
# TODO: Set the CXX standard here as well.
set(CMAKE_C_STANDARD_REQUIRED ON)

option(ENABLE_OPENCL "Build OpenCL backend" ON)
option(ENABLE_CUDA "Build Cuda backend" OFF)
option(ENABLE_SYCL "Build SYCL backend" OFF)

option(SYCL_IMPLEMENTATION "Specify the SYCL implementation package")

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

set(SOURCES src/nomp.c src/lpy.c src/aux.c src/log.c src/mem.c src/jit.c)
if (ENABLE_OPENCL)
  list(APPEND SOURCES src/opencl.c)
  add_compile_definitions(OPENCL_ENABLED)
endif()
if (ENABLE_CUDA)
  list(APPEND SOURCES src/cuda.c)
  add_compile_definitions(CUDA_ENABLED)
endif()
if (ENABLE_SYCL)
  list(APPEND SOURCES src/sycl.cpp)
  add_compile_definitions(SYCL_ENABLED)
endif()
add_library(nomp SHARED ${SOURCES})

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
target_link_libraries(nomp PRIVATE Python3::Python)
find_package(OpenSSL REQUIRED)
target_link_libraries(nomp PRIVATE OpenSSL::SSL)
if (ENABLE_OPENCL)
  find_package(OpenCL REQUIRED)
  target_link_libraries(nomp PRIVATE OpenCL::OpenCL)
endif()
if (ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(nomp PRIVATE CUDA::cudart CUDA::nvrtc)
endif()
if (ENABLE_SYCL)
  find_package(${SYCL_IMPLEMENTATION} REQUIRED)
  target_link_libraries(nomp PRIVATE ${SYCL_IMPLEMENTATION_LIBRARIES})
endif()

set_target_properties(nomp PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION 1
  PUBLIC_HEADER src/nomp.h)
target_include_directories(nomp PRIVATE src)

option(ENABLE_DOCS "Enable Documentation" OFF)
if (ENABLE_DOCS)
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")
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
