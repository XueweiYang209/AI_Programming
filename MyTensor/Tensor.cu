#include "Tensor.h"

// GPU上对value初始化
__global__ void initKernel(float* data, size_t size){
    CUDA_KERNEL_LOOP(i, size){
        data[i] = 0; //初始化为0
    }
}

Tensor::Tensor(const std::vector<int>& shape, Device device) 
    : shape_(shape), device_(device) {
    total_size_ = 1;
    for (auto it = shape_.begin(); it != shape_.end();it++) {
        total_size_ *= *it;
    }
    allocateMemory();
}

Tensor::~Tensor() {}

void Tensor::allocateMemory(){
    if (device_ == Device::CPU) {
        value = std::shared_ptr<float[]>(new float[total_size_]);
        std::fill(value.get(), value.get() + total_size_, 0);
    } else if (device_ == Device::GPU) {
        float* gpuPtr;
        cudaMalloc(&gpuPtr, total_size_ * sizeof(float));
        initKernel<<<CudaGetBlocks(total_size_),kCudaThreadsNum>>>(gpuPtr, total_size_);
        cudaDeviceSynchronize();
        value = std::shared_ptr<float[]>(gpuPtr, [](float* ptr) { cudaFree(ptr); }); // 使用自定义删除器管理 GPU 内存
    } else {
        throw std::invalid_argument("Unsupported device");
    }
}

Tensor Tensor::cpu(){
    Tensor t(this->shape_,Device::CPU);
    if (this->device_ == Device::CPU){
        memcpy(t.value.get(),this->value.get(),total_size_ * sizeof(float)); //深拷贝
    }
    else {
        cudaMemcpy(t.value.get(), this->value.get(), total_size_ * sizeof(float), cudaMemcpyDeviceToHost);
    }
    return t;
}

Tensor Tensor::gpu(){
    Tensor t(this->shape_,Device::GPU);
    if (this->device_ == Device::CPU){
            cudaMemcpy(t.value.get(), this->value.get(), total_size_ * sizeof(float), cudaMemcpyHostToDevice);
        }
    else {
        cudaMemcpy(t.value.get(), this->value.get(), total_size_ * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    return t;
}
