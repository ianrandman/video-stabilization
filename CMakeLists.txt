cmake_minimum_required(VERSION 3.0)
project(project_1)

find_package(OpenCV REQUIRED)
#include_directories(${OpenCV_INCLUDE-DIRECTORIES})
add_executable(main_cv dad9489_ixr5487.cpp)

#set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)

target_link_libraries(main_cv ${OpenCV_LIBS})