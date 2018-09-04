cmake_minimum_required(VERSION 2.8.3)
project(coin_detector)

add_definitions(-std=c++11)

find_package(OpenCV REQUIRED)

## Specify additional locations of header files
include_directories(
	${OpenCV_INCLUDE_DIRS}

)

## Declare a C++ executable
add_executable(coin_detector src/detector.cpp)

## Specify libraries to link a library or executable target against
target_link_libraries(coin_detector
  ${OpenCV_LIBRARIES}
)
