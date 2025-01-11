cmake_minimum_required(VERSION 3.15)

project(ModularCNN LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

find_package(pybind11 REQUIRED)

pybind11_add_module(ModularCNN MODULE pybind/bindings.cpp)