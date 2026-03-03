cmake_minimum_required(VERSION 3.14)
project(chessengine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --------------------------------------------------
# Compiler warnings and optimizations
# --------------------------------------------------
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# --------------------------------------------------
# Find pybind11 (installed via pip or system package manager)
# --------------------------------------------------
find_package(pybind11 REQUIRED)

# --------------------------------------------------
# Build the Python extension module
# --------------------------------------------------
pybind11_add_module(chessengine
    chessengine/attack_tables.cpp
    chessengine/bindings.cpp
    chessengine/board.cpp
    chessengine/movegen.cpp
    chessengine/utils.cpp
)

target_include_directories(chessengine
    PRIVATE ${PROJECT_SOURCE_DIR}/chessengine
)

# --------------------------------------------------
# Installation
# --------------------------------------------------
install(TARGETS chessengine
    LIBRARY DESTINATION .
)