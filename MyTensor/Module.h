#ifndef MODULE_H
#define MODULE_H

#include "Tensor.h"
#include <cublas_v2.h>

// 卷积网络算子
void gemm_gpu(cublasOperation_t trans1,cublasOperation_t trans2,int m,int k,int n,
    float alpha,float* input,float* weight,float beta,float* output); // gpu上矩阵乘
void forward_fc(Tensor input,Tensor& output,Tensor weight,int batch_size,int in_features,int out_features);
void backward_fc(Tensor input,Tensor weight,int batch_size,int in_features,int out_features,
Tensor grad_output,Tensor& grad_input,Tensor& grad_weight);

void im2col(float* image,float* column,int batch_size,int in_features,int height,int width);
void col2im(float* grad_column, float* grad_input, int batch_size, int in_features, int height, int width);
void forward_conv(Tensor input,Tensor& column,Tensor weight,Tensor& output,int batch_size,int in_features,int out_features,int height,int width);
void backward_conv(Tensor column, Tensor &grad_column, Tensor weight,int batch_size, int in_features, int out_features,
int height, int width, Tensor grad_output,Tensor &grad_input, Tensor &grad_weight);

void maxpool(float* in_data,float* out_data,float* mask,int batch_size,int in_features,int height,int width);
void forward_maxpool(Tensor input, Tensor &output, Tensor &mask,int batch_size,int in_features,int height,int width);
void unpool(float* out_data,float* in_data,float* mask,int batch_size,int in_features,int height,int width);
void backward_maxpool(Tensor grad_output, Tensor &grad_input, Tensor mask,int batch_size,int in_features,int height,int width);

void softmax(float* input, float* output, int batch_size, int in_features);
void forward_softmax(Tensor input, Tensor &output, int batch_size, int in_features);

void forward_cross_entropy(Tensor input, Tensor &loss, Tensor label, int batch_size, int in_features);
void backward_cross_entropy(Tensor input, Tensor &grad_loss, Tensor label, int batch_size, int in_features);

void Relu(Tensor in, Tensor& out);
void Relu_backward(Tensor Loss_grad_out, Tensor& Loss_grad_in, Tensor in);
void Sigmoid(Tensor in, Tensor& out);
void Sigmoid_backward(Tensor Loss_grad_out, Tensor &Loss_grad_in, Tensor out);

// 张量算子
Tensor EWiseAdd(Tensor &a, Tensor &b);
Tensor AddScalar(Tensor &a, float scalar);
Tensor EWiseMul(Tensor &a, Tensor &b);
Tensor MulScalar(Tensor &a, float scalar);
Tensor PowerScalar(Tensor &a, float scalar);
Tensor EWisePow(Tensor &a, Tensor &b);
Tensor EWiseDiv(Tensor &a, Tensor &b);
Tensor DivScalar(Tensor &a, float scalar);
Tensor Negate(Tensor &input);
Tensor Log(Tensor &input);
Tensor Exp(Tensor &input);
Tensor Matmul(Tensor &a, Tensor &b);
Tensor Reshape(Tensor a, const std::vector<int> &shape);

#endif // MODULE_H