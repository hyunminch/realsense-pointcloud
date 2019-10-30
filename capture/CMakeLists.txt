cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(capture)

add_executable(capture capture.cpp)

find_package(PCL 1.8 REQUIRED)
SET(CMAKE_CXX_FLAGS "-std=c++11")
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

target_link_libraries(capture ${PCL_LIBRARIES} librealsense2.so ${OpenCV_LIBS})
