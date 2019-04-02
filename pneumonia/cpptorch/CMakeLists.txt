cmake_minimum_required(VERSION 3.0)

set(CMAKE_PREFIX_PATH /Users/adamkowalski/pytorch/torch)

project(cpptorch)

find_package(Boost REQUIRED COMPONENTS filesystem)
find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

add_executable(${PROJECT_NAME} src/main.cc)

target_include_directories(cpptorch PRIVATE ${OpenCV_INCLUDE_DIRS})

target_link_libraries(cpptorch PRIVATE ${OpenCV_LIBS})
target_link_libraries(cpptorch PRIVATE ${TORCH_LIBRARIES})
target_link_libraries(cpptorch PRIVATE ${Boost_LIBRARIES})

target_compile_features(cpptorch PRIVATE cxx_std_17)
