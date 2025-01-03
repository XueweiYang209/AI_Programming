#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <vector>
#include <memory>
#include <stdexcept>

const int kCudaThreadsNum = 512;
inline int CudaGetBlocks(const int N){
    return (N + kCudaThreadsNum - 1) / kCudaThreadsNum;
}

#define CUDA_KERNEL_LOOP(i,n) \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

enum class Device {
    CPU,
    GPU
};


class Tensor {
public:
    Tensor(const std::vector<int>& shape, Device device);
    ~Tensor();

    Tensor cpu();
    Tensor gpu();
    float* data() { return value.get();};
    int length() const { return total_size_; };
    void Print(){
        for (int i = 0; i < this->total_size_; i++){
            std::cout << this->value.get()[i] << " ";
        }
        std::cout<< std::endl;
    }
    void assign(const std::vector<float>& values) {
        for (size_t i = 0; i < total_size_ && i < values.size(); ++i) {
            value[i] = values[i];
        }
    }
    std::vector<int> shape_;

private:
    std::shared_ptr<float[]> value; // 使用 shared_ptr 管理内存
    Device device_;
    size_t total_size_;

    void allocateMemory(); // 分配内存
};

#endif // TENSOR_H