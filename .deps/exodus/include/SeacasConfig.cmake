# @HEADER
# *****************************************************************************
#            TriBITS: Tribal Build, Integrate, and Test System
#
# Copyright 2013-2016 NTESS and the TriBITS contributors.
# SPDX-License-Identifier: BSD-3-Clause
# *****************************************************************************
# @HEADER

get_filename_component(_Seacas_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
get_filename_component(_Seacas_PREFIX "${_Seacas_DIR}" PATH)
set(Seacas_DIR "${_Seacas_PREFIX}/lib/cmake/Seacas")
message(WARNING "SeacasConfig.cmake has moved.  "
  "It now exists at a location under the installation prefix where the "
  "find_package command looks by default (<prefix>/lib/cmake/Seacas).  "
  "This compatibility file exists at the old location (<prefix>/include) "
  "to present this message and load the file from its new location."
  "\n"
  "The find_package() call that loaded this file did so because its "
  "cached result variable, Seacas_DIR, is set to\n"
  "  ${_Seacas_DIR}\n"
  "I'm locally setting Seacas_DIR to\n"
  "  ${Seacas_DIR}\n"
  "and loading SeacasConfig.cmake from its new location.  "
  "One may suppress this warning by setting the above value in the cache.  "
  "However, the application needs modification permanently fix the issue.  "
  "The find_package() call that loaded this file may have the form\n"
  "  find_package(Seacas REQUIRED PATHS \${Seacas_PATH}/include)\n"
  "Change it to the form\n"
  "  set(CMAKE_PREFIX_PATH \${Seacas_PATH} \${CMAKE_PREFIX_PATH})\n"
  "  find_package(Seacas REQUIRED)\n"
  "to find SeacasConfig.cmake in its new location in future builds "
  "while still honoring the Seacas_PATH option for this application."
  )
include(${Seacas_DIR}/SeacasConfig.cmake)
