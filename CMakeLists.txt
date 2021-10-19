cmake_minimum_required(VERSION 3.9)

project(libgnomp VERSION 0.0.1 DESCRIPTION "gnomp runtime library" LANGUAGES C)

include(GNUInstallDirs)

add_library(gnomp SHARED src/init.c)
set_target_properties(gnomp PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION 1
  PUBLIC_HEADER src/gnomp.h)
target_include_directories(gnomp PRIVATE src)
configure_file(gnomp.pc.in gnomp.pc @ONLY)

find_package(OpenCL REQUIRED)
target_link_libraries(gnomp OpenCL::OpenCL)

install(TARGETS gnomp
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_PUBLIC_INCLUDEDIR})
install(FILES ${CMAKE_BINARY_DIR}/gnomp.pc
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/pkgconfig)
