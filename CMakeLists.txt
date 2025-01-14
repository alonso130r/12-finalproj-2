cmake_minimum_required(VERSION 3.15)
include(FetchContent)

project(ModularCNN LANGUAGES CXX)

set(CMAKE_C_COMPILER "/MinGW/bin/gcc.exe")
set(CMAKE_CXX_COMPILER "C:/Users/adamk/AppData/Local/JetBrains/CLion 2024.3.1.1/bin/mingw/bin/g++.exe")

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

set(PYBIND11_FINDPYTHON ON)
FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1  # Latest stable version
)
FetchContent_MakeAvailable(pybind11)


pybind11_add_module(ModularCNN MODULE pybind/bindings.cpp)