# Copyright (C) 2019 - 2022 by the deal.II authors
#
# This file is part of the deal.II library.
#
# The deal.II library is free software; you can use it, redistribute
# it, and/or modify it under the terms of the GNU Lesser General
# Public License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
# The full text of the license can be found in the file LICENSE at
# the top level of the deal.II distribution.

macro(set_if_empty _variable)
  if("${${_variable}}" STREQUAL "")
    set(${_variable} ${ARGN})
  endif()
endmacro()

set(SYMENGINE_DIR
    ""
    CACHE PATH "An optional hint to a SymEngine installation")
set_if_empty(SYMENGINE_DIR "$ENV{SYMENGINE_DIR}")

set(_cmake_module_path ${CMAKE_MODULE_PATH})

find_package(
  SymEngine
  CONFIG
  QUIET
  HINTS
  ${SYMENGINE_DIR}
  PATH_SUFFIXES
  lib/cmake/symengine
  NO_SYSTEM_ENVIRONMENT_PATH)

string(REGEX
       REPLACE "(lib64|lib)\\/cmake\\/symengine\\/\\.\\.\\/\\.\\.\\/\\.\\.\\/"
               "" SYMENGINE_INCLUDE_DIRS "${SYMENGINE_INCLUDE_DIRS}")

if(SymEngine_FOUND)
  add_library(nomp::SymEngine INTERFACE IMPORTED)
  set_target_properties(
    nomp::SymEngine
    PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${SYMENGINE_INCLUDE_DIRS}"
               INTERFACE_LINK_LIBRARIES "${SYMENGINE_LIBRARIES}")
endif()
