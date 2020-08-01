cmake_minimum_required(VERSION 3.16)
project(NeuralBoy)

set(CMAKE_CXX_STANDARD 14)


include_directories(/Users/huangege/cpp_libs/eigen-3.3.7)

add_executable(NeuralBoy main.cpp net_helper.cpp activations.cpp data_helper.cpp activations.h data_helper.h net_helper.h)
#add_executable(NeuralBoy main.cpp file_helper.cpp)
