cmake_minimum_required(VERSION 2.8)


get_filename_component(PROJNAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
Project(${PROJNAME})
Message(STATUS "-------------------------------")
Message(STATUS "Processing Project ${PROJNAME}:")

# offer the choice of having shared_sources as a sub-folder... good for packaging a sample
# if BASE_DIRECTORY not defined, it means this cmake file was called as the first entry point and not included
if(NOT BASE_DIRECTORY) # if not defined, it means this cmake file was called as the first entry point and not included
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/shared_sources)
    Message(STATUS "found sub-folder shared_sources")
    SET(BASE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    INCLUDE(${CMAKE_CURRENT_SOURCE_DIR}/shared_sources/CMakeLists_include.txt)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/shared_sources ${CMAKE_BINARY_DIR}/shared_sources)
  elseif(EXISTS $ENV{NVPRO_ROOT_DIR}/shared_sources)
    FILE(TO_CMAKE_PATH "$ENV{NVPRO_ROOT_DIR}" BASE_DIRECTORY)
    INCLUDE(${BASE_DIRECTORY}/shared_sources/CMakeLists_include.txt)
    add_subdirectory(${BASE_DIRECTORY}/shared_sources ${CMAKE_BINARY_DIR}/shared_sources)
  elseif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../../shared_sources)
    SET(BASE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../..)
    INCLUDE(${CMAKE_CURRENT_SOURCE_DIR}/../../shared_sources/CMakeLists_include.txt)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../shared_sources ${CMAKE_BINARY_DIR}/shared_sources)
  else()
    SET(BASE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..)
    INCLUDE(${CMAKE_CURRENT_SOURCE_DIR}/../shared_sources/CMakeLists_include.txt)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../shared_sources ${CMAKE_BINARY_DIR}/shared_sources)
  endif()
endif()
_add_project_definitions(${PROJNAME})

if (SUPPORT_NVTOOLSEXT)
add_definitions(-DSUPPORT_NVTOOLSEXT)
endif(SUPPORT_NVTOOLSEXT)

#####################################################################################
# additions from packages needed for this sample
# add refs  in LIBRARIES_OPTIMIZED
# add refs  in LIBRARIES_DEBUG
# add files in PACKAGE_SOURCE_FILES
#

_add_package_Cuda () 
if( NOT CUDA_FOUND )
  message(SEND_ERROR "Baking sample requires CUDA")
endif()

_add_package_Optix ()
if ( NOT OPTIX_FOUND )
  message(SEND_ERROR "Baking sample requires OptiX")
endif()

# Find Eigen template lib for least squares fitting.
# If any other samples need this library in the future, then move this block
# to higher level CMakeLists.txt.

set( EIGEN3_PATH "/usr/include/eigen3" CACHE PATH "Path to optional Eigen3 dir, for least squares filtering in baking sample" )

if ( EXISTS "${EIGEN3_PATH}/signature_of_eigen3_matrix_library" )
  set(EIGEN3_ENABLED TRUE)
else()
  set( EIGEN3_ENABLED FALSE )
endif()

if (EIGEN3_ENABLED)
  include_directories( ${EIGEN3_PATH} )
  add_definitions(-DEIGEN3_ENABLED=1)
endif()

# end Eigen


#####################################################################################
# Source files for this project
#
file(GLOB SOURCE_FILES *.cpp *.hpp *.inl *.h *.c)
file(GLOB GLSL_FILES *.glsl)
file(GLOB CUDA_FILES *.cu)


#####################################################################################
# Executable
#
if(WIN32 AND NOT GLUT_FOUND)
  add_definitions(/wd4996) #remove printf warning
  add_definitions(/wd4244) #remove double to float conversion warning
  add_definitions(/wd4305) #remove double to float truncation warning
  cuda_add_executable(${PROJNAME} WIN32 ${SOURCE_FILES} ${COMMON_SOURCE_FILES} ${PACKAGE_SOURCE_FILES} ${GLSL_FILES} ${CUDA_FILES})
else()
  cuda_add_executable(${PROJNAME} ${SOURCE_FILES} ${COMMON_SOURCE_FILES} ${PACKAGE_SOURCE_FILES} ${GLSL_FILES} ${CUDA_FILES})
endif()

#####################################################################################
# common source code needed for this sample
#
source_group(common FILES 
  ${COMMON_SOURCE_FILES}
  ${PACKAGE_SOURCE_FILES}
)
source_group(shaders FILES 
  ${GLSL_FILES}
)
source_group(CUDA FILES 
  ${CUDA_FILES}
)

#####################################################################################
# Linkage
#

# Need winmm for timing code on Windows
if (WIN32) 
  LIST(APPEND PLATFORM_LIBRARIES winmm.lib)
endif()

target_link_libraries(${PROJNAME} optimized
    ${LIBRARIES_OPTIMIZED}
    ${PLATFORM_LIBRARIES}
    shared_sources
)
target_link_libraries(${PROJNAME} debug
    ${LIBRARIES_DEBUG}
    ${PLATFORM_LIBRARIES}
    shared_sources
)

#####################################################################################
# copies binaries that need to be put next to the exe files (ZLib, etc.)
#

_copy_binaries_to_target( ${PROJNAME} )

