#include "Module.h"

void gemm_gpu(cublasOperation_t trans1, cublasOperation_t trans2, int m, int k,
              int n, float alpha, float *A, float *B, float beta, float *C) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    const float *alf = &alpha;
    const float *bet = &beta;
    // cublasSgemm为列优先，下面通过领先维度和转置变量的修改，达到行优先输出的效果
    int lda = trans1 == CUBLAS_OP_N ? k : m;
    int ldb = trans2 == CUBLAS_OP_N ? n : k;
    int ldc = m;
    trans1 = trans1 == CUBLAS_OP_N ? CUBLAS_OP_T : CUBLAS_OP_N;
    trans2 = trans2 == CUBLAS_OP_N ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasSgemm(handle, trans1, trans2, m, n, k, alf, A, lda, B, ldb, bet, C,
                ldc); // 矩阵乘的结果在行优先逻辑下仿佛做了一次转置
}

void forward_fc(Tensor input, Tensor &output, Tensor weight, int batch_size,
                int in_features, int out_features) {
    // 公式两端同时取转置，再进行矩阵乘，以抵消gemm_gpu对输出的转置效果，下同
    gemm_gpu(CUBLAS_OP_N, CUBLAS_OP_T, out_features, in_features, batch_size,
             1.0, weight.data(), input.data(), 0.0, output.data());
}
void backward_fc(Tensor input, Tensor output, Tensor weight, int batch_size,
                 int in_features, int out_features, Tensor grad_output,
                 Tensor &grad_input, Tensor &grad_weight) {
    gemm_gpu(CUBLAS_OP_T, CUBLAS_OP_T, in_features, out_features, batch_size,
             1.0, weight.data(), grad_output.data(), 0.0, grad_input.data());
    gemm_gpu(CUBLAS_OP_T, CUBLAS_OP_N, in_features, batch_size, out_features,
             1.0, input.data(), grad_output.data(), 0.0, grad_weight.data());
}

__global__ void im2col_kernel(const float *image, float *column,
                              int in_features, int height, int width) {
    int h = blockIdx.x;  // 当前高度索引
    int w = threadIdx.x; // 当前宽度索引

    int output_row = h * width + w; // 输出行在 column 中的位置
    int filter_size = 3;            // 卷积核大小
    int pad = 1;                    // zero padding 大小

    // 遍历各个通道
    for (int c = 0; c < in_features; ++c) {
        for (int i = 0; i < filter_size; ++i) {
            for (int j = 0; j < filter_size; ++j) {
                int image_h = h - pad + i;
                int image_w = w - pad + j;

                int output_col =
                    c * (filter_size * filter_size) + i * filter_size + j;

                // 检查是否越界
                if (image_h >= 0 && image_h < height && image_w >= 0 &&
                    image_w < width) {
                    column[output_row * (9 * in_features) + output_col] =
                        image[(c * height + image_h) * width + image_w];
                } else {
                    column[output_row * (9 * in_features) + output_col] = 0.0f;
                }
            }
        }
    }
}

void im2col(float *image, float *column, int batch_size, int in_features,
            int height, int width) {
    for (int b = 0; b < batch_size; ++b) {
        im2col_kernel<<<height, width>>>(
            image + b * in_features * height * width,
            column + b * height * width * 9 * in_features, in_features, height,
            width);
    }
    cudaDeviceSynchronize();
}

__global__ void col2im_kernel(const float *grad_column, float *grad_input,
                              int in_features, int height, int width) {
    // 每一个线程处理grad_column的一整行向量
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // 确保索引在范围内
    if (index < height * width) {
        int w_col = index % width; // 对应 grad_input 中的宽度索引
        int h_col = index / width; // 对应 grad_input 中的高度索引

        // 对应的输入图像位置
        for (int c = 0; c < in_features; ++c) {
            for (int i = 0; i < 3; ++i) { // 对应 3x3 卷积核
                for (int j = 0; j < 3; ++j) {
                    int h_in = h_col + i - 1; // 因为是 3x3 卷积，行偏移
                    int w_in = w_col + j - 1; // 列偏移

                    // 确保在合法范围内
                    if (h_in >= 0 && h_in < height && w_in >= 0 &&
                        w_in < width) {
                        // grad_column 在内存中的位置
                        int col_index =
                            index * in_features * 9 + c * 9 + i * 3 + j;
                        atomicAdd(&grad_input[c * height * width +
                                              h_in * width + w_in],
                                  grad_column[col_index]);
                    }
                }
            }
        }
    }
}

void col2im(float *grad_column, float *grad_input, int batch_size,
            int in_features, int height, int width) {
    for (int b = 0; b < batch_size; ++b) {
        // 计算当前图像的 grad_column 起始位置
        const float *grad_col_ptr =
            grad_column + b * height * width * in_features * 9;
        float *grad_input_ptr = grad_input + b * in_features * height * width;

        col2im_kernel<<<CudaGetBlocks(height * width), kCudaThreadsNum>>>(
            grad_col_ptr, grad_input_ptr, in_features, height, width);
    }

    cudaDeviceSynchronize();
}

void forward_conv(Tensor input, Tensor &column, Tensor weight, Tensor &output,
                  int batch_size, int in_features, int out_features, int height,
                  int width) {
    // 先做ima2col，再进行矩阵乘
    im2col(input.data(), column.data(), batch_size, in_features, height, width);
    for (int i = 0; i < batch_size; ++i) {
        gemm_gpu(CUBLAS_OP_N, CUBLAS_OP_T, height * width, 9 * in_features,
                 out_features, 1,
                 column.data() + i * 9 * in_features * height * width,
                 weight.data(), 0,
                 output.data() + i * out_features * height * width);
    }
}
void backward_conv(Tensor column, Tensor &grad_column, Tensor weight,
                   int batch_size, int in_features, int out_features,
                   int height, int width, Tensor grad_output,
                   Tensor &grad_input, Tensor &grad_weight) {
    // 先进行矩阵乘，再col2im
    for (int i = 0; i < batch_size; ++i) {
        gemm_gpu(CUBLAS_OP_T, CUBLAS_OP_T, in_features * 9, height * width,
                 out_features, 1,
                 column.data() + i * 9 * in_features * height * width,
                 grad_output.data() + i * out_features * height * width, 1,
                 grad_weight.data());
        gemm_gpu(CUBLAS_OP_T, CUBLAS_OP_N, in_features * 9, out_features,
                 height * width, 1, weight.data(),
                 grad_output.data() + i * out_features * height * width, 0,
                 grad_column.data() + i * 9 * in_features * height * width);
    }
    col2im(grad_column.data(), grad_input.data(), batch_size, in_features,
           height, width);
}

__global__ void maxpool_kernel(float *in_data, float *out_data, float *mask,
                               int batch_size, int in_features, int height,
                               int width) {
    int b = blockIdx.x; // 当前处理的图像 batch 索引
    int c = blockIdx.y; // 当前处理的特征通道索引

    int h_out = height / 2; // 使用2x2的池化窗口
    int w_out = width / 2;

    int h = threadIdx.x; // 输出的高度索引
    int w = threadIdx.y; // 输出的宽度索引

    // 初始化最大值和对应的输入位置
    int max_index = b * in_features * height * width + c * height * width +
                    h * 2 * width + w * 2;
    float max_value = in_data[max_index];
    // 遍历池化窗口
    for (int ph = 0; ph < 2; ++ph) {
        for (int pw = 0; pw < 2; ++pw) {
            int h_in = h * 2 + ph; // 计算输入位置
            int w_in = w * 2 + pw;
            // 计算输入数据在一维数组中的索引
            if (h_in < height && w_in < width) { // 确保不越界
                int in_index = b * in_features * height * width +
                               c * height * width + h_in * width + w_in;
                // 更新最大值
                if (in_data[in_index] > max_value) {
                    max_value = in_data[in_index];
                    max_index = in_index;
                }
            }
        }
    }

    // 将最大值写入输出数据
    int out_index =
        b * in_features * h_out * w_out + c * h_out * w_out + h * w_out + w;
    out_data[out_index] = max_value;

    // 记录最大值的输入位置
    mask[max_index] = 1.0f;
}
void maxpool(float *in_data, float *out_data, float *mask, int batch_size,
             int in_features, int height, int width) {
    dim3 blocks(batch_size, in_features);
    dim3 threads(height / 2, width / 2);

    maxpool_kernel<<<blocks, threads>>>(in_data, out_data, mask, batch_size,
                                        in_features, height, width);
    cudaDeviceSynchronize();
}
void forward_maxpool(Tensor input, Tensor &output, Tensor &mask, int batch_size,
                     int in_features, int height, int width) {
    maxpool(input.data(), output.data(), mask.data(), batch_size, in_features,
            height, width);
}

__global__ void unpool_kernel(float *out_data, float *in_data, float *mask,
                              int batch_size, int in_features, int height,
                              int width) {
    // 每个线程处理一个out_data中的像素
    int b = blockIdx.x;
    int c = blockIdx.y;
    int h = threadIdx.x;
    int w = threadIdx.y;
    int h_out = height / 2;
    int w_out = width / 2;
    // 计算out_data中像素及in_data池化窗口左上角像素对应的索引
    int out_index =
        b * in_features * h_out * w_out + c * h_out * w_out + h * w_out + w;
    int in_index = b * in_features * height * width + c * height * width +
                   h * 2 * width + w * 2;
    in_data[in_index] = out_data[out_index] * mask[in_index];
    in_data[in_index + 1] = out_data[out_index] * mask[in_index + 1];
    in_data[in_index + width] = out_data[out_index] * mask[in_index + width];
    in_data[in_index + width + 1] =
        out_data[out_index] * mask[in_index + width + 1];
}
void unpool(float *out_data, float *in_data, float *mask, int batch_size,
            int in_features, int height, int width) {
    dim3 blocks(batch_size, in_features);
    dim3 threads(height / 2, width / 2);

    unpool_kernel<<<blocks, threads>>>(out_data, in_data, mask, batch_size,
                                       in_features, height, width);
    cudaDeviceSynchronize();
}
void backward_maxpool(Tensor grad_output, Tensor &grad_input, Tensor mask,
                      int batch_size, int in_features, int height, int width) {
    unpool(grad_output.data(), grad_input.data(), mask.data(), batch_size,
           in_features, height, width);
}

__global__ void compute_max(const float *input, float *max_vals,
                            int in_features, int batch_size) {
    // 一个线程计算一行数据的最大值
    int row = blockIdx.x;
    if (row < batch_size) {
        float max_val = -INFINITY;
        for (int j = 0; j < in_features; ++j) {
            float val = input[row * in_features + j];
            if (val > max_val) {
                max_val = val;
            }
        }
        max_vals[row] = max_val;
    }
}
__global__ void subtract_and_exponentiate(float *input, float *output,
                                          const float *max_vals,
                                          int in_features, int batch_size) {
    // 每个线程处理一个元素，减去最大值后指数化
    int row = blockIdx.x;
    int col = threadIdx.x;

    if (row < batch_size && col < in_features) {
        output[row * in_features + col] =
            exp(input[row * in_features + col] - max_vals[row]);
    }
}
__global__ void reduce_sum_kernel(const float *input, float *sum_vals,
                                  int in_features, int batch_size) {
    // 每个线程计算一行的元素和
    int row = blockIdx.x;
    if (row < batch_size) {
        float sum = 0.0f;
        for (int j = 0; j < in_features; j++) {
            sum += input[row * in_features + j];
        }
        sum_vals[row] = sum;
    }
}
__global__ void normalize_kernel(float *input, float *sum_vals, int in_features,
                                 int batch_size) {
    // 每个线程对一个元素进行归一化
    int row = blockIdx.x;
    int col = threadIdx.x;
    if (row < batch_size && col < in_features) {
        input[row * in_features + col] /= sum_vals[row];
    }
}

void softmax(float *input, float *output, int batch_size, int in_features) {
    float *d_max_vals, *d_sum_vals;

    cudaMalloc(&d_max_vals, batch_size * sizeof(float));
    cudaMalloc(&d_sum_vals, batch_size * sizeof(float));

    compute_max<<<batch_size, 1>>>(input, d_max_vals, in_features, batch_size);
    cudaDeviceSynchronize();

    subtract_and_exponentiate<<<batch_size, in_features>>>(
        input, output, d_max_vals, in_features, batch_size);
    cudaDeviceSynchronize();

    reduce_sum_kernel<<<batch_size, 1>>>(output, d_sum_vals, in_features,
                                         batch_size);
    cudaDeviceSynchronize();

    normalize_kernel<<<batch_size, in_features>>>(output, d_sum_vals,
                                                  in_features, batch_size);
    cudaDeviceSynchronize();

    cudaFree(d_max_vals);
    cudaFree(d_sum_vals);
}
void forward_softmax(Tensor input, Tensor &output, int batch_size,
                     int in_features) {
    softmax(input.data(), output.data(), batch_size, in_features);
}

__global__ void cross_entropy_kernel(float *input, float *loss, float *label,
                                     int batch_size, int in_features) {
    // 找到真实标签对应的索引，计算交叉熵损失
    int row = threadIdx.x;
    loss[row] = -log(input[row * in_features + int(label[row])]);
}
void forward_cross_entropy(Tensor input, Tensor &loss, Tensor label,
                           int batch_size, int in_features) {
    cross_entropy_kernel<<<1, batch_size>>>(
        input.data(), loss.data(), label.data(), batch_size, in_features);
}
__global__ void derivative_loss_kernel(float *input, float *label,
                                       float *grad_loss, int batch_size,
                                       int in_features) {
    // 计算交叉熵损失的梯度
    int row = blockIdx.x;
    int col = threadIdx.x;
    if (col == int(label[row])) {
        grad_loss[row * in_features + col] = input[row * in_features + col] - 1;
    } else {
        grad_loss[row * in_features + col] = input[row * in_features + col];
    }
}
void backward_cross_entropy(Tensor input, Tensor &grad_loss, Tensor label,
                            int batch_size, int in_features) {
    derivative_loss_kernel<<<batch_size, in_features>>>(
        input.data(), label.data(), grad_loss.data(), batch_size, in_features);
}

__global__ void relu(float *in, float *out, int size) {
    CUDA_KERNEL_LOOP(i, size) { out[i] = (in[i] > 0) ? in[i] : 0; }
}
void Relu(Tensor in, Tensor &out) {
    int size = in.length();
    relu<<<CudaGetBlocks(size), kCudaThreadsNum>>>(in.data(), out.data(), size);
}

__global__ void relu_backward(float *Loss_grad_out, float *Loss_grad_in,
                              float *in, int size) {
    CUDA_KERNEL_LOOP(i, size) {
        Loss_grad_in[i] = (in[i] > 0) ? Loss_grad_out[i] : 0;
    }
}
void Relu_backward(Tensor Loss_grad_out, Tensor &Loss_grad_in, Tensor in) {
    int size = Loss_grad_out.length();
    relu_backward<<<CudaGetBlocks(size), kCudaThreadsNum>>>(
        Loss_grad_out.data(), Loss_grad_in.data(), in.data(), size);
}

__global__ void sigmoid(float *in, float *out, int size) {
    CUDA_KERNEL_LOOP(i, size) { out[i] = 1 / (1 + exp(-in[i])); }
}
void Sigmoid(Tensor in, Tensor &out) {
    int size = in.length();
    sigmoid<<<size, kCudaThreadsNum>>>(in.data(), out.data(), size);
}

__global__ void sigmoid_backward(float *Loss_grad_out, float *Loss_grad_in,
                                 float *out, int size) {
    CUDA_KERNEL_LOOP(i, size) {
        Loss_grad_in[i] = Loss_grad_out[i] * out[i] * (1 - out[i]);
    }
}
void Sigmoid_backward(Tensor Loss_grad_out, Tensor &Loss_grad_in, Tensor out) {
    int size = Loss_grad_out.length();
    sigmoid_backward<<<CudaGetBlocks(size), kCudaThreadsNum>>>(
        Loss_grad_out.data(), Loss_grad_in.data(), out.data(), size);
}

__global__ void ewise_add_kernel(const float *a, const float *b, float *output,
                                 int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = a[idx] + b[idx];
    }
}

void EWiseAdd(Tensor &a, Tensor &b, Tensor &output) {
    int size = a.length();
    ewise_add_kernel<<<CudaGetBlocks(size), kCudaThreadsNum>>>(
        a.data(), b.data(), output.data(), size);
    cudaDeviceSynchronize();
}

__global__ void add_scalar_kernel(const float *a, float scalar, float *output,
                                  int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = a[idx] + scalar;
    }
}

void AddScalar(Tensor &a, float scalar, Tensor &output) {
    int size = a.length();
    add_scalar_kernel<<<CudaGetBlocks(size), kCudaThreadsNum>>>(
        a.data(), scalar, output.data(), size);
    cudaDeviceSynchronize();
}

__global__ void ewise_mul_kernel(const float *a, const float *b, float *output,
                                 int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = a[idx] * b[idx];
    }
}

void EWiseMul(Tensor &a, Tensor &b, Tensor &output) {
    int size = a.length();
    ewise_mul_kernel<<<CudaGetBlocks(size), kCudaThreadsNum>>>(
        a.data(), b.data(), output.data(), size);
    cudaDeviceSynchronize();
}

__global__ void mul_scalar_kernel(const float *a, float scalar, float *output,
                                  int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = a[idx] * scalar;
    }
}

void MulScalar(Tensor &a, float scalar, Tensor &output) {
    int size = a.length();
    mul_scalar_kernel<<<CudaGetBlocks(size), kCudaThreadsNum>>>(
        a.data(), scalar, output.data(), size);
    cudaDeviceSynchronize();
}

__global__ void power_scalar_kernel(const float *a, float scalar, float *output,
                                    int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = powf(a[idx], scalar);
    }
}

void PowerScalar(Tensor &a, float scalar, Tensor &output) {
    int size = a.length();
    power_scalar_kernel<<<CudaGetBlocks(size), kCudaThreadsNum>>>(
        a.data(), scalar, output.data(), size);
    cudaDeviceSynchronize();
}

__global__ void ewise_pow_kernel(const float *a, const float *b, float *output,
                                 int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = powf(a[idx], b[idx]);
    }
}

void EWisePow(Tensor &a, Tensor &b, Tensor &output) {
    int size = a.length();
    ewise_pow_kernel<<<CudaGetBlocks(size), kCudaThreadsNum>>>(
        a.data(), b.data(), output.data(), size);
    cudaDeviceSynchronize();
}

__global__ void ewise_div_kernel(const float *a, const float *b, float *output,
                                 int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = a[idx] / b[idx];
    }
}

void EWiseDiv(Tensor &a, Tensor &b, Tensor &output) {
    int size = a.length();
    ewise_div_kernel<<<CudaGetBlocks(size), kCudaThreadsNum>>>(
        a.data(), b.data(), output.data(), size);
    cudaDeviceSynchronize();
}

__global__ void div_scalar_kernel(const float *a, float scalar, float *output,
                                  int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = a[idx] / scalar;
    }
}

void DivScalar(Tensor &a, float scalar, Tensor &output) {
    int size = a.length();
    div_scalar_kernel<<<CudaGetBlocks(size), kCudaThreadsNum>>>(
        a.data(), scalar, output.data(), size);
    cudaDeviceSynchronize();
}

__global__ void negate_kernel(const float *input, float *output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = -input[idx];
    }
}

void Negate(Tensor &input, Tensor &output) {
    int size = input.length();
    negate_kernel<<<CudaGetBlocks(size), kCudaThreadsNum>>>(
        input.data(), output.data(), size);
    cudaDeviceSynchronize();
}

__global__ void log_kernel(const float *input, float *output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = logf(input[idx]);
    }
}

void Log(Tensor &input, Tensor &output) {
    int size = input.length();
    log_kernel<<<CudaGetBlocks(size), kCudaThreadsNum>>>(input.data(),
                                                         output.data(), size);
    cudaDeviceSynchronize();
}

__global__ void exp_kernel(const float *input, float *output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = expf(input[idx]);
    }
}

void Exp(Tensor &input, Tensor &output) {
    int size = input.length();
    exp_kernel<<<CudaGetBlocks(size), kCudaThreadsNum>>>(input.data(),
                                                         output.data(), size);
    cudaDeviceSynchronize();
}