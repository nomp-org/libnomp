cmake_minimum_required(VERSION 3.9)

project(libnomp VERSION 0.0.1 DESCRIPTION "nomp runtime library" LANGUAGES C)

include(GNUInstallDirs)

add_library(nomp SHARED src/opencl.c src/nomp.c)
set_target_properties(nomp PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION 1
  PUBLIC_HEADER src/nomp.h)
target_include_directories(nomp PRIVATE src)
configure_file(nomp.pc.in nomp.pc @ONLY)

find_package(OpenCL REQUIRED)
target_link_libraries(nomp OpenCL::OpenCL)

add_executable(vec-init tests/vec-init.c)
target_link_libraries(vec-init nomp)
target_include_directories(vec-init PUBLIC src)

install(TARGETS nomp
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_PUBLIC_INCLUDEDIR})
install(FILES ${CMAKE_BINARY_DIR}/nomp.pc
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/pkgconfig)
