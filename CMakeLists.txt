cmake_minimum_required(VERSION 3.16)
project(gstnvavmstitch LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED
  gstreamer-1.0
  gstreamer-base-1.0
  gstreamer-video-1.0)
find_package(CUDA 11.4 REQUIRED)

find_path(NVBUF_INCLUDE_DIR nvbufsurface.h
  PATHS /usr/src/jetson_multimedia_api/include
  NO_DEFAULT_PATH)
find_library(NVBUF_SURFACE_LIBRARY nvbufsurface
  PATHS /usr/lib/aarch64-linux-gnu/tegra /usr/lib/aarch64-linux-gnu
  NO_DEFAULT_PATH)

if(NOT NVBUF_INCLUDE_DIR OR NOT NVBUF_SURFACE_LIBRARY)
  message(FATAL_ERROR "Jetson NvBufSurface headers/library not found")
endif()

add_library(gstnvavmstitch SHARED
  src/gstnvavmstitch.cpp
  src/avm_asset.cpp
  src/avm_stitch.cu)

target_include_directories(gstnvavmstitch PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${NVBUF_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIRS}
  ${GSTREAMER_INCLUDE_DIRS})
target_link_directories(gstnvavmstitch PRIVATE ${GSTREAMER_LIBRARY_DIRS})
target_link_libraries(gstnvavmstitch PRIVATE
  ${GSTREAMER_LIBRARIES}
  ${CUDA_CUDART_LIBRARY}
  ${CUDA_CUDA_LIBRARY}
  ${NVBUF_SURFACE_LIBRARY}
  EGL)
target_compile_definitions(gstnvavmstitch PRIVATE
  PACKAGE="gstnvavmstitch"
  PACKAGE_NAME="gstnvavmstitch"
  GST_USE_UNSTABLE_API)
set_target_properties(gstnvavmstitch PROPERTIES
  PREFIX lib
  POSITION_INDEPENDENT_CODE ON)
target_compile_options(gstnvavmstitch PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:${GSTREAMER_CFLAGS_OTHER}>
  $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_72,code=sm_72>
  $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_87,code=sm_87>)

pkg_get_variable(GSTREAMER_SYSTEM_PLUGIN_DIR gstreamer-1.0 pluginsdir)
set(GSTREAMER_PLUGIN_DIR "${GSTREAMER_SYSTEM_PLUGIN_DIR}"
  CACHE PATH "GStreamer plugin directory")
install(TARGETS gstnvavmstitch LIBRARY DESTINATION ${GSTREAMER_PLUGIN_DIR})
