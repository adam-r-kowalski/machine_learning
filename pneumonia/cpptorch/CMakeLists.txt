cmake_minimum_required(VERSION 3.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

project(cpptorch)

add_subdirectory(lib/pybind11)

find_package(Torch REQUIRED)

add_executable(cpptorch src/main.cc)

target_link_libraries(cpptorch PRIVATE pybind11::embed)

target_link_libraries(cpptorch PRIVATE ${TORCH_LIBRARIES}) 

target_compile_features(cpptorch PRIVATE cxx_std_17)
