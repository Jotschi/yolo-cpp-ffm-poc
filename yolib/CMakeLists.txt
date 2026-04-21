cmake_minimum_required(VERSION 3.16)
project(yolo_ort)

option(ONNXRUNTIME_DIR "Path to built ONNX Runtime directory." STRING)
message(STATUS "ONNXRUNTIME_DIR: ${ONNXRUNTIME_DIR}")

set(YOLOS_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../YOLOs-CPP-1.0.0" CACHE PATH "Path to YOLOs-CPP v1.0.0 source directory")
if(NOT EXISTS "${YOLOS_CPP_DIR}/include/yolos/yolos.hpp")
    set(YOLOS_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../YOLOs-CPP")
endif()

if(NOT EXISTS "${YOLOS_CPP_DIR}/include/yolos/yolos.hpp")
    message(FATAL_ERROR "Could not find YOLOs-CPP headers. Set -DYOLOS_CPP_DIR=/path/to/YOLOs-CPP-v1.0.0")
endif()

message(STATUS "YOLOS_CPP_DIR: ${YOLOS_CPP_DIR}")

find_package(OpenCV REQUIRED)

include_directories("${YOLOS_CPP_DIR}/include")

add_library(yolib SHARED
    src/yolib.cpp)

set_target_properties(yolib PROPERTIES VERSION "1.0.0")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

target_include_directories(yolib PUBLIC "${ONNXRUNTIME_DIR}/include")
target_compile_features(yolib PRIVATE cxx_std_17)
target_link_libraries(yolib ${OpenCV_LIBS})
target_link_libraries(yolib "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")

if(APPLE)
set_target_properties(yolib PROPERTIES
         LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../src/main/resources/native/macosx)
endif()

if(UNIX AND NOT APPLE)
set_target_properties(yolib PROPERTIES
         LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../src/main/resources/native/linux)
endif()