cmake_minimum_required(VERSION 3.16)
project(traces_backprop)

set(CMAKE_CXX_STANDARD 17)

add_executable(traces_backprop main.cpp src/NeuralNetwork.cpp include/NeuralNetwork.h src/Synapse.cpp src/Neuron.cpp)
