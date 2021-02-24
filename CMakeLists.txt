cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
cmake_policy(SET CMP0074 NEW)
project(PyTorchToCpp)

#set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -O3")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

#find_package(Caffe2 REQUIRED)
#find_package(Torch REQUIRED)
#
#
add_executable(GAN src/GAN.cpp)
#target_link_libraries(wGAN "${TORCH_LIBRARIES}" nvidia-ml)
set_property(TARGET GAN PROPERTY CXX_STANDARD 17)
#
#add_executable(BIBAE src/BIBAE.cpp src/util.cpp src/util.h)
#target_link_libraries(BIBAE "${TORCH_LIBRARIES}" nvidia-ml)
#set_property(TARGET BIBAE PROPERTY CXX_STANDARD 17)
