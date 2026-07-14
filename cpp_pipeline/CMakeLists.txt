cmake_minimum_required(VERSION 3.10)
project(perception_cpp_pipeline CXX)

# Enforce C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find OpenCV components
find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d dnn highgui)
message(STATUS "Found OpenCV: ${OpenCV_VERSION}")
message(STATUS "OpenCV Include Dirs: ${OpenCV_INCLUDE_DIRS}")

# Find Eigen3 headers (support both standard package check and header fallback paths)
find_path(EIGEN3_INCLUDE_DIR NAMES Eigen/Dense PATHS /usr/include/eigen3 /usr/local/include/eigen3)
if(NOT EIGEN3_INCLUDE_DIR)
    message(FATAL_ERROR "Eigen3 headers not found. Please install libeigen3-dev.")
else()
    message(STATUS "Found Eigen3 headers: ${EIGEN3_INCLUDE_DIR}")
endif()

# Find ONNX Runtime headers and libraries
find_path(ONNXRUNTIME_INCLUDE_DIR NAMES onnxruntime_c_api.h PATH_SUFFIXES onnxruntime PATHS /usr/include /usr/local/include)
find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime PATHS /usr/lib/x86_64-linux-gnu /usr/local/lib)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR "ONNX Runtime not found. Please install libonnxruntime-dev.")
else()
    message(STATUS "Found ONNX Runtime headers: ${ONNXRUNTIME_INCLUDE_DIR}")
    message(STATUS "Found ONNX Runtime library: ${ONNXRUNTIME_LIBRARY}")
endif()

# Build the core pipeline static library
add_library(perception_pipeline STATIC
    src/vision_pipeline.cpp
    src/ekf.cpp
)

# Set include directories
target_include_directories(perception_pipeline PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    ${EIGEN3_INCLUDE_DIR}
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

# Link library dependencies
target_link_libraries(perception_pipeline PUBLIC
    ${OpenCV_LIBRARIES}
    ${ONNXRUNTIME_LIBRARY}
    pthread
)

# Build verification test executable
add_executable(verify_pipeline
    src/verify_pipeline.cpp
)

target_link_libraries(verify_pipeline PRIVATE
    perception_pipeline
)

# Build production node executable
add_executable(perception_node
    src/main.cpp
)

target_link_libraries(perception_node PRIVATE
    perception_pipeline
)
