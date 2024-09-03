cmake_minimum_required(VERSION 3.0)
project(SemanticallyGuidedFeatureMatching)
# c++ version

SET(CMAKE_BUILD_TYPE Debug)

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

add_definitions(-w)

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

message("Opencv DIR: " ${OpenCV_DIR})

include_directories(
   ${PROJECT_SOURCE_DIR}
   ${PROJECT_SOURCE_DIR}/include
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} SHARED
src/ORBextractor.cc
src/SPextractor.cc
src/SIFTextractor.cc
src/SuperPoint.cc
src/Frame.cc
src/Initializer.cc
)

target_link_libraries(${PROJECT_NAME}
   ${OpenCV_LIBS}
   ${TORCH_LIBRARIES}
)

message("Opencv Libs: " ${OpenCV_LIBS})

# Build examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
add_executable(main
main.cc)
target_link_libraries(main ${PROJECT_NAME}) 
set_property(TARGET main PROPERTY CXX_STANDARD 17)





