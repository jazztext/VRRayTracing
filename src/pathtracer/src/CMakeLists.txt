cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)

set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)
set(BUILD_SHARED_LIBS ON)

set(CUDA_BUILD_CUBIN ON)

set(CUDA_NVCC_FLAGS
   ${CUDA_NVCC_FLAGS};
   -gencode arch=compute_20,code=sm_35 -rdc=true
)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3 -lineinfo -maxrregcount 64)

set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPOGATE_HOST_FLAGS OFF)

set(CUDA_SOURCE
    gpu/bbox.cu
    gpu/bvhGPU.cu
    gpu/bsdf.cu
    gpu/matrix3x3.cu
    gpu/matrix4x4.cu
    gpu/pathtracer.cu
    gpu/primitiveGPU.cu
    gpu/vector4D.cu
    gpu/sampler.cu
    gpu/light.cu
    gpu/rayBenchmark.cu
    camera.cu
)

# Application source
set(APPLICATION_SOURCE

    # Collada Parser
    collada/collada.cpp
    collada/camera_info.cpp
    collada/light_info.cpp
    collada/sphere_info.cpp
    collada/polymesh_info.cpp
    collada/material_info.cpp

    # Static scene
    static_scene/sphere.cpp
    static_scene/triangle.cpp
    static_scene/object.cpp
    #static_scene/environment_light.cpp

    # PathTracer
    bvh.cpp
    sampler.cpp
    bbox.cpp
    halfEdgeMesh.cpp

    # Application
    application.cpp
)

#-------------------------------------------------------------------------------
# Set include directories
#-------------------------------------------------------------------------------
include_directories(
  ${GLEW_INCLUDE_DIRS}
  ${GLFW_INCLUDE_DIRS}
  ${CMU462_INCLUDE_DIRS}
  ${FREETYPE_INCLUDE_DIRS}
)

#-------------------------------------------------------------------------------
# Set link directories
#-------------------------------------------------------------------------------
link_directories(
  ${GLEW_LIBRARY_DIRS}
  ${GLFW_LIBRARY_DIRS}
  ${CMU462_LIBRARY_DIRS}
  ${FREETYPE_LIBRARY_DIRS}
)

#-------------------------------------------------------------------------------
# Add executable
#-------------------------------------------------------------------------------
cuda_add_library(pathtracer STATIC ${APPLICATION_SOURCE} ${CUDA_SOURCE})

target_link_libraries( pathtracer
    CMU462 ${CMU462_LIBRARIES}
    glew ${GLEW_LIBRARIES}
    glfw ${GLFW_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${FREETYPE_LIBRARIES}
    ${CMAKE_THREADS_INIT}
)

cuda_compile_ptx(ptx_files ${CUDA_SOURCE})

add_custom_target(ptx ALL DEPENDS ${ptx_files} SOURCES ${CUDA_SOURCE})

#-------------------------------------------------------------------------------
# Platform-specific configurations for target
#-------------------------------------------------------------------------------
if(APPLE)
  set_property( TARGET pathtracer APPEND_STRING PROPERTY COMPILE_FLAGS
                "-Wno-deprecated-declarations -Wno-c++11-extensions")
endif(APPLE)

# Put executable in build directory root
set(LIBRARY_OUTPUT_PATH ../../..)

# Install to project root
install(TARGETS pathtracer ARCHIVE DESTINATION ${PROJECT_SOURCE_DIR})
