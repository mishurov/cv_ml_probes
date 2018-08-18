cmake_minimum_required(VERSION 3.7)
project(face)

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

# training
add_executable(train sample_train_landmark_detector2.cpp)
target_link_libraries(train ${OpenCV_LIBRARIES})

# demo
set(SOURCES_TRAIN
    landmarks.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/common.cpp
)

#add_definitions(-DDEBUG)

add_executable(landmarks ${SOURCES_TRAIN})

set_target_properties(landmarks PROPERTIES COMPILE_FLAGS
    "${CMAKE_CXX_FLAGS} -std=c++11"
)

target_link_libraries(landmarks
    ${OpenCV_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${GLEW_LIBRARIES}
)
