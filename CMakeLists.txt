cmake_minimum_required(VERSION 3.14)
project(chessengine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --------------------------------------------------
# Instead of FetchContent, just add the cloned subdir
# --------------------------------------------------
add_subdirectory(${PROJECT_SOURCE_DIR}/extern/pybind11)

# --------------------------------------------------
# Build the Python extension
# --------------------------------------------------
pybind11_add_module(chessengine
    chessengine/board.cpp
    chessengine/movegen.cpp
    chessengine/utils.cpp
    chessengine/bindings.cpp
    chessengine/attack_tables.cpp
)

target_include_directories(chessengine
    PRIVATE ${PROJECT_SOURCE_DIR}/chessengine
)
