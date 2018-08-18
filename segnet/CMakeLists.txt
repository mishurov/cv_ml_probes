cmake_minimum_required(VERSION 3.7)
project(ar)

find_package(OpenCV 3.4.1 EXACT REQUIRED PATHS
    ${CMAKE_CURRENT_SOURCE_DIR}/../deps/opencv/
)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../deps/tensorflow/include
)

link_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../deps/tensorflow/lib
)

set(SOURCES predict.cpp)

add_executable(predict ${SOURCES})

set_target_properties(predict PROPERTIES COMPILE_FLAGS
    "${CMAKE_CXX_FLAGS} -std=c++11"
)

target_link_libraries(predict
    tensorflow_cc
    tensorflow_framework
    ${OpenCV_LIBRARIES}
)
