cmake_minimum_required(VERSION 3.22)
project(libnomp VERSION 0.0.1 DESCRIPTION "nomp runtime library" LANGUAGES C CXX)

# Nomp options.
option(ENABLE_OPENCL "Build OpenCL backend" OFF)
option(ENABLE_CUDA "Build CUDA backend" OFF)
option(ENABLE_HIP "Build HIP backend" OFF)
option(ENABLE_DOCS "Enable Documentation" OFF)
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)

# Nomp compile time constants which goes in `nomp-defs.h`.
set(NOMP_MAX_BUFFER_SIZE 128)
set(NOMP_MAX_SOURCE_SIZE 16384)
set(NOMP_MAX_CFLAGS_SIZE 16384)
set(NOMP_MAX_KERNEL_ARGS_SIZE 64)
set(NOMP_MAX_SCRATCH_SIZE 65536)
set(NOMP_DEFAULT_VERBOSE 2)
set(NOMP_DEFAULT_PROFILE 0)
set(NOMP_DEFAULT_DEVICE 0)
set(NOMP_DEFAULT_PLATFORM 0)
configure_file(include/nomp-defs.h.in include/nomp-defs.h @ONLY)

# C standard options.
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Set cmake module path.
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")

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

if (ENABLE_DOCS)
  add_subdirectory ("docs")
endif()

set(SOURCES src/nomp.c src/loopy.c src/symengine.c src/aux.c src/log.c
  src/reduction.c)
if (ENABLE_OPENCL)
  find_package(OpenCL REQUIRED)
  if (OpenCL_FOUND)
    list(APPEND SOURCES backends/opencl.c)
  else()
    message(FATAL_ERROR "ENABLE_OPENCL is ON but unable to find OPENCL runtime.")
  endif()
endif()
if (ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  if (CUDAToolkit_FOUND)
    list(APPEND SOURCES backends/cuda.c)
  else()
    message(FATAL_ERROR "ENABLE_CUDA is ON but unable to find CUDA runtime.")
  endif()
endif()
if (ENABLE_HIP)
  find_package(HIP)
  if(HIP_FOUND)
    list(APPEND SOURCES backends/hip.c)
  else()
    message(FATAL_ERROR "ENABLE_HIP is ON but unable to find HIP runtime.")
  endif()
endif()

add_library(nomp SHARED ${SOURCES})
set_target_properties(nomp PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION 1
  PUBLIC_HEADER include/nomp.h)
target_include_directories(nomp PRIVATE include ${CMAKE_BINARY_DIR}/include)
target_compile_options(nomp PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
target_link_libraries(nomp PRIVATE Python3::Python)

find_package(OpenSSL REQUIRED)
target_link_libraries(nomp PRIVATE OpenSSL::SSL)

find_package(SymEngine REQUIRED)
target_include_directories(nomp PRIVATE ${SYMENGINE_INCLUDE_DIRS})
target_link_libraries(nomp PRIVATE ${SYMENGINE_LIBRARIES})

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

# Tests
file(GLOB TESTS tests/nomp-*.c)
foreach(test ${TESTS})
  string(REPLACE "${CMAKE_SOURCE_DIR}/tests/" "" temp ${test})
  string(REPLACE ".c" "" test_exe ${temp})
  add_executable(${test_exe} ${test})
  target_link_libraries(${test_exe} nomp)
  target_include_directories(${test_exe} PRIVATE include tests)
  target_compile_options(${test_exe} PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>)
  if (HAS_ADDRESS_SANITIZER)
    target_compile_options(${test_exe} PRIVATE -fsanitize=address)
    target_link_options(${test_exe} PRIVATE -fsanitize=address)
  endif()
  install (TARGETS ${test_exe} RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/tests)
endforeach()

# Add AddressSanitizer if it is enabled and supported.
set(HAS_ADDRESS_SANITIZER FALSE)
if (ENABLE_ASAN)
  include(CheckLinkerFlag)
  check_linker_flag(C -fsanitize=address ASAN_SUPPORTED)
  set(HAS_ADDRESS_SANITIZER ${ASAN_SUPPORTED})
  if (NOT HAS_ADDRESS_SANITIZER)
    message(FATAL_ERROR "AddressSanitizer is enabled but not supported.")
  else()
    target_compile_options(nomp PRIVATE -fsanitize=address)
    target_link_options(nomp PRIVATE -fsanitize=address)
  endif()
endif()

# Add clang-format as a custom target if available.
find_program(CLANG_FORMAT NAMES clang-format)
if (CLANG_FORMAT)
  add_custom_target(format
    COMMAND ${CLANG_FORMAT} -i ${CMAKE_SOURCE_DIR}/*/*.[ch]
    COMMENT "Running clang-format ...")
  add_custom_target(format-check
    COMMAND ${CLANG_FORMAT} --dry-run --Werror -i ${CMAKE_SOURCE_DIR}/*/*.[ch]
    COMMENT "Running clang-format check ...")
endif()

# Add clang-tidy as a custom target if available.
find_program(CLANG_TIDY NAMES clang-tidy)
if (CLANG_TIDY)
  add_custom_target(tidy
    COMMAND ${CLANG_TIDY} -p ${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR}/*/*.[ch]
    COMMENT "Running clang-tidy ...")
endif()

# Add pylint as a custom target if available.
find_program(PYLINT NAMES pylint)
if (PYLINT)
  add_custom_target(pylint
    COMMAND ${PYLINT} ${CMAKE_SOURCE_DIR} --recursive yes --rcfile=${CMAKE_SOURCE_DIR}/.pylintrc
    COMMENT "Running pylint ...")
endif()

# Add black as a custom target if available.
find_program(BLACK NAMES black)
if (BLACK)
  add_custom_target(black
    COMMAND ${BLACK} ${CMAKE_CURRENT_SOURCE_DIR}/
    COMMENT "Running black ...")
  add_custom_target(black-check
    COMMAND ${BLACK} --check ${CMAKE_CURRENT_SOURCE_DIR}/
    COMMENT "Running black check ...")
endif()

# Add isort as a custom target if available.
find_program(ISORT NAMES isort)
if (ISORT)
  add_custom_target(isort
    COMMAND ${ISORT} ${CMAKE_CURRENT_SOURCE_DIR}/
    COMMENT "Running isort ...")
  add_custom_target(isort-check
    COMMAND ${ISORT} --check-only ${CMAKE_CURRENT_SOURCE_DIR}/
    COMMENT "Running isort check ...")
endif()

# Add flake8 as a custom target if available.
find_program(FLAKE8 NAMES flake8)
if (FLAKE8)
  add_custom_target(flake8
    COMMAND ${FLAKE8} ${CMAKE_CURRENT_SOURCE_DIR}/
    COMMENT "Running flake8 ...")
endif()

install(TARGETS nomp LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/python DESTINATION ${CMAKE_INSTALL_PREFIX})

install(DIRECTORY tests/ DESTINATION ${CMAKE_INSTALL_PREFIX}/tests
  FILES_MATCHING PATTERN "*.py")

install(DIRECTORY scripts/ DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
  FILE_PERMISSIONS OWNER_READ OWNER_EXECUTE OWNER_WRITE
  GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

configure_file(nomp.pc.in nomp.pc @ONLY)
install(FILES ${CMAKE_BINARY_DIR}/nomp.pc DESTINATION
  ${CMAKE_INSTALL_PREFIX}/lib/pkgconfig)
