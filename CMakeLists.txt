cmake_minimum_required(VERSION 3.10)
project(rs-pcl)

set(CMAKE_CXX_STANDARD 14)

add_executable(rs-pcl ./src/main.cpp)

find_package(realsense2)
find_package(OpenCV REQUIRED)
find_package(OpenGL REQUIRED)
find_package(PCL 1.9 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

set(DEPENDENCIES realsense2 ${OpenCV_LIBS} glfw ${OPENGL_LIBRARIES} ${PCL_LIBRARIES})

target_link_libraries(${PROJECT_NAME} PRIVATE ${DEPENDENCIES})
