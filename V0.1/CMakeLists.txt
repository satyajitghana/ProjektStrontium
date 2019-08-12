cmake_minimum_required(VERSION 3.0.0)
project(ProjektStrontium VERSION 0.1.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

include(CTest)
enable_testing()

include_directories(src)

add_executable(ProjektStrontium main.cpp NeuralNet.cpp NeuralNet.hpp Neuron.cpp Neuron.hpp)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
