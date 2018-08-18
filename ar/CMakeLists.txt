cmake_minimum_required(VERSION 3.7)
project(ar)

find_package(OpenCV 3.4.1 EXACT REQUIRED PATHS
    ${CMAKE_CURRENT_SOURCE_DIR}/../deps/opencv/
)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${OPENGL_INCLUDE_DIR}
    ${GLEW_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/
)


set(SOURCES
    augreal.cpp
    #proto.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/common.cpp
)

add_executable(augreal ${SOURCES})

set_target_properties(augreal PROPERTIES COMPILE_FLAGS
    "${CMAKE_CXX_FLAGS} -std=c++11"
)

target_link_libraries(augreal
    ${OpenCV_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${GLEW_LIBRARIES}
)
