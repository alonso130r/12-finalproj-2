cmake_minimum_required(VERSION 3.31)

project(ModularCNN LANGUAGES CXX)

set(CMAKE_C_COMPILER "/opt/homebrew/Cellar/gcc/14.2.0_1/bin/gcc-14")
set(CMAKE_CXX_COMPILER "/opt/homebrew/Cellar/gcc/14.2.0_1/bin/g++-14")

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libgcc -static-libstdc++")

set(PYBIND11_FINDPYTHON ON)

find_package(OpenMP REQUIRED)
find_package(pybind11 REQUIRED)
target_link_libraries(... OpenMP::OpenMP_CXX)

pybind11_add_module(ModularCNN MODULE layers/ConvolutionLayer.h layers/ConvolutionLayer.tpp layers/FullyConnectedLayer.h layers/FullyConnectedLayer.tpp layers/Layer.h layers/Layer.tpp layers/MaxPoolingLayer.h layers/MaxPoolingLayer.tpp tools/AMSGrad.h tools/AMSGrad.tpp tools/ComputationGraph.h tools/ComputationGraph.tpp tools/ConnectedWeights.h tools/ConnectedWeights.tpp tools/ConvolutionalWeights.h tools/ConvolutionalWeights.tpp tools/ConvolutionOperation.h tools/ConvolutionOperation.tpp tools/CrossEntropy.h tools/CrossEntropy.tpp tools/FullyConnectedOperation.h tools/FullyConnectedOperation.tpp tools/LayerConfig.h tools/LayerConfig.cpp tools/MaxPoolingOperation.h tools/MaxPoolingOperation.tpp tools/Operation.h tools/Operation.cpp tools/PoolingWeights.h tools/PoolingWeights.tpp tools/Tensor.h tools/Tensor.tpp tools/WeightStruct.h tools/WeightStruct.cpp model/ModularCNN.h model/ModularCNN.tpp pybind/bindings.cpp)

if(APPLE)
    set_target_properties(ModularCNN PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
endif(APPLE)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(ModularCNN PRIVATE -fdiagnostics-color=always)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(ModularCNN PRIVATE -fcolor-diagnostics)
endif()