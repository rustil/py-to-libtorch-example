cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
#cmake_policy(SET CMP0074 NEW)
project(PyTorchToCpp)

set(CMAKE_VERBOSE_MAKEFILE ON)

## To run with rustil/qu-pytorch:base from DockerHub
## !! To be removed and set accordingly otherwise
set(CMAKE_PREFIX_PATH /opt/conda/lib/python3.8/site-packages/torch/share/cmake/Torch/)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -O3")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(Torch REQUIRED)


add_executable(GAN src/GAN.cpp)
target_link_libraries(GAN "${TORCH_LIBRARIES}" )
set_property(TARGET GAN PROPERTY CXX_STANDARD 17)
