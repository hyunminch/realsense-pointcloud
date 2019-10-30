cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(PCDFileVisualization)

find_package(PCL 1.8 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(pcdVisualization pcdVisualization.cpp)
target_link_libraries(pcdVisualization ${PCL_LIBRARIES})

add_executable(cloudViewer cloudViewer.cpp)
target_link_libraries(cloudViewer ${PCL_LIBRARIES})
