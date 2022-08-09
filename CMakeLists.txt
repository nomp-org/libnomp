cmake_minimum_required(VERSION 3.22)
project(libnomp VERSION 0.0.1 DESCRIPTION "nomp runtime library" LANGUAGES C)
include(GNUInstallDirs)

option(ENABLE_OPENCL "Build OpenCL backend" ON)
option(ENABLE_CUDA "Build Cuda backend" OFF)

set(SOURCES src/nomp.c src/loopy.c src/auxiliary.c)
if (ENABLE_OPENCL)
  list(APPEND SOURCES src/opencl.c)
  add_compile_definitions(OPENCL_ENABLED)
endif()
if (ENABLE_CUDA)
  list(APPEND SOURCES src/cuda.c)
  enable_language(CUDA)
  add_compile_definitions(CUDA_ENABLED)
endif()
add_library(nomp SHARED ${SOURCES})

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
target_link_libraries(nomp PRIVATE Python3::Python)
if (ENABLE_OPENCL)
  find_package(OpenCL REQUIRED)
  target_link_libraries(nomp PRIVATE OpenCL::OpenCL)
endif()

set_target_properties(nomp PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION 1
  PUBLIC_HEADER src/nomp.h)
target_include_directories(nomp PRIVATE src)

configure_file(nomp.pc.in nomp.pc @ONLY)

file(GLOB TESTS tests/nomp-api-*.c)
foreach(test ${TESTS})
  string(REPLACE "${CMAKE_SOURCE_DIR}/tests/" "" temp ${test})
  string(REPLACE ".c" "" test_exe ${temp})
  add_executable(${test_exe} ${test})
  target_link_libraries(${test_exe} nomp)
  target_include_directories(${test_exe} PUBLIC src)
endforeach(test ${TESTS})

install(TARGETS nomp
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES ${CMAKE_BINARY_DIR}/nomp.pc
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/pkgconfig)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/python
  DESTINATION ${CMAKE_INSTALL_PREFIX})
