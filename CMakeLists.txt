cmake_minimum_required(VERSION 3.15)

project(ModularCNN LANGUAGES CXX)

set(CMAKE_C_COMPILER "/opt/homebrew/Cellar/gcc/14.2.0_1/bin/gcc-14")
set(CMAKE_CXX_COMPILER "/opt/homebrew/Cellar/gcc/14.2.0_1/bin/g++-14")

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)

pybind11_add_module(ModularCNN MODULE pybind/bindings.cpp)