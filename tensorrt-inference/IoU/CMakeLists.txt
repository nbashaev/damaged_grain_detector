cmake_minimum_required(VERSION 3.2)
project(IoU)

find_package(CUDA QUIET REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost 1.54 REQUIRED filesystem system)
list(APPEND CUDA_NVCC_FLAGS "-std=c++11 ")
add_definitions(-std=c++11)
set(SRC IoU.cpp)
set(SRC_H)

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
	/home/nikita/TensorRT-5.0.2.6/include
    )

set(CMAKE_CXX_STANDARD 11)

add_executable(IoU ${SRC} ${SRC_H})
target_link_libraries(IoU
        ${OpenCV_LIBS}
        ${CUDA_LIBRARIES}
        ${Boost_LIBRARIES}
	/home/nikita/TensorRT-5.0.2.6/targets/x86_64-linux-gnu/lib/libnvinfer.so
    )
