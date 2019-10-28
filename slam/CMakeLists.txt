cmake_minimum_required(VERSION 3.15)
project(slam)

set(CMAKE_CXX_STANDARD 14)

add_executable(slam main.cpp)

find_package(realsense2)
find_package(OpenGL REQUIRED)
find_package(PCL 1.8 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

set(DEPENDENCIES realsense2 glfw ${OPENGL_LIBRARIES} ${PCL_LIBRARIES})

target_link_libraries(${PROJECT_NAME} PRIVATE ${DEPENDENCIES})
