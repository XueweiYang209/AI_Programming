cmake_minimum_required(VERSION 3.22)
project(MyTensor LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(pybind11 REQUIRED)

# 添加 pybind11 模块
pybind11_add_module(MyTensor binding.cpp Tensor.cu Module.cu)

# 链接库
target_link_libraries(MyTensor PRIVATE CUDA::cublas CUDA::cudart)
target_include_directories(MyTensor PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

# 启用 CUDA 特性
set_target_properties(MyTensor PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
