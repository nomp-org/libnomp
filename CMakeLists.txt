cmake_minimum_required(VERSION 3.22)

project(libnomp VERSION 0.0.1 DESCRIPTION "nomp runtime library" LANGUAGES C)

include(GNUInstallDirs)

add_library(nomp SHARED src/opencl.c src/nomp.c src/loopy.c src/auxiliary.c)
set_target_properties(nomp PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION 1
  PUBLIC_HEADER src/nomp.h)
target_include_directories(nomp PRIVATE src)
configure_file(nomp.pc.in nomp.pc @ONLY)

find_package(OpenCL REQUIRED)
target_link_libraries(nomp PRIVATE OpenCL::OpenCL)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
target_link_libraries(nomp PRIVATE Python3::Python)

file(GLOB TESTS tests/nomp-api-*.c)
foreach(test ${TESTS})
  string(REPLACE "${CMAKE_SOURCE_DIR}/tests/" "" temp ${test})
  string(REPLACE ".c" "" test_exe ${temp})
  add_executable(${test_exe} ${test})
  target_link_libraries(${test_exe} nomp)
  target_include_directories(${test_exe} PUBLIC src)
endforeach(test ${TESTS})

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
add_subdirectory ("docs")

install(TARGETS nomp
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_PUBLIC_INCLUDEDIR})
install(FILES ${CMAKE_BINARY_DIR}/nomp.pc
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/pkgconfig)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/python
  DESTINATION ${CMAKE_INSTALL_PREFIX})
