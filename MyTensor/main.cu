#include "Module.h"
#include "Tensor.h"
#include <iostream>

// 以下函数通过在cpu串行实现对应module功能，用于测试

// 串行实现全连接层
void forward_fc_serial(Tensor input, Tensor weight, Tensor output) {
    for (int i = 0; i < output.shape_[0]; i++) {
        for (int j = 0; j < output.shape_[1]; j++) {
            for (int k = 0; k < input.shape_[1]; k++) {
                output.data()[i * output.shape_[1] + j] +=
                    input.data()[i * input.shape_[1] + k] *
                    weight.data()[j * weight.shape_[1] + k];
            }
        }
    }
}
void backward_fc_serial(Tensor input, Tensor output, Tensor weight,
                        int batch_size, int in_features, int out_features,
                        Tensor grad_output, Tensor &grad_input,
                        Tensor &grad_weight) {
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < in_features; j++) {
            for (int k = 0; k < out_features; k++) {
                grad_input.data()[i * in_features + j] +=
                    grad_output.data()[i * out_features + k] *
                    weight.data()[k * in_features + j];
            }
        }
    }
    for (int i = 0; i < out_features; i++) {
        for (int j = 0; j < in_features; j++) {
            for (int k = 0; k < batch_size; k++) {
                grad_weight.data()[i * in_features + j] +=
                    grad_output.data()[k * out_features + i] *
                    input.data()[k * in_features + j];
            }
        }
    }
}

// 串行实现卷积层
void forward_conv_serial(Tensor input, Tensor weight, Tensor output,
                         int batch_size, int in_features, int out_features,
                         int height, int width) {
    // 假设 weight 的形状为 (out_features, in_features, kernel_height,
    // kernel_width)
    int kernel_height = 3; // 假设卷积核高度为 3
    int kernel_width = 3;  // 假设卷积核宽度为 3

    // 输出特征图的高度和宽度
    int output_height = height; // 如果没有填充的话，输出尺寸将与输入相同
    int output_width = width;

    // 遍历每个批次
    for (int b = 0; b < batch_size; ++b) {
        // 遍历每个输出特征
        for (int out = 0; out < out_features; ++out) {
            // 遍历输出特征图的每一个位置
            for (int h = 0; h < output_height; ++h) {
                for (int w = 0; w < output_width; ++w) {
                    float sum = 0.0f; // 初始化卷积和
                    // 遍历输入特征图的每一个通道
                    for (int in = 0; in < in_features; ++in) {
                        // 遍历卷积核的每一个位置
                        for (int kh = 0; kh < kernel_height; ++kh) {
                            for (int kw = 0; kw < kernel_width; ++kw) {
                                // 计算输入特征图的对应位置
                                int input_h =
                                    h + kh -
                                    kernel_height / 2; // 考虑卷积核的中心
                                int input_w =
                                    w + kw -
                                    kernel_width / 2; // 考虑卷积核的中心

                                // 确保输入索引在合法范围内
                                if (input_h >= 0 && input_h < height &&
                                    input_w >= 0 && input_w < width) {
                                    // 计算对应的权重索引
                                    int weight_index =
                                        out * (in_features * kernel_height *
                                               kernel_width) +
                                        in * (kernel_height * kernel_width) +
                                        kh * kernel_width + kw;
                                    // 计算输入索引
                                    int input_index =
                                        b * (in_features * height * width) +
                                        in * (height * width) +
                                        input_h * width + input_w;

                                    sum += input.data()[input_index] *
                                           weight.data()
                                               [weight_index]; // 累加卷积结果
                                }
                            }
                        }
                    }
                    // 将计算结果存入输出
                    int output_index =
                        b * (out_features * output_height * output_width) +
                        out * (output_height * output_width) +
                        h * output_width + w;
                    output.data()[output_index] = sum;
                    // 将最终的卷积和赋值到输出
                }
            }
        }
    }
}

void backward_conv_serial(Tensor column, Tensor weight, int batch_size,
                          int in_features, int out_features, int height,
                          int width, Tensor grad_output, Tensor grad_input,
                          Tensor grad_weight) {
    int filter_size = 3; // 卷积核大小
    int pad = 1;         // zero padding 大小

    // 计算 grad_weight
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                for (int co = 0; co < out_features; ++co) {
                    for (int c = 0; c < in_features;
                         ++c) { // 添加 in_features 循环
                        for (int i = 0; i < filter_size; ++i) {
                            for (int j = 0; j < filter_size; ++j) {
                                int column_h = h * width + w; // column 的行索引
                                grad_weight
                                    .data()[co * (9 * in_features) +
                                            c * (filter_size * filter_size) +
                                            i * filter_size + j] +=
                                    column.data()
                                        [(b * (height * width) + column_h) *
                                             (9 * in_features) +
                                         c * (filter_size * filter_size) +
                                         i * filter_size + j] *
                                    grad_output.data()[(b * (height * width) +
                                                        h * width + w) *
                                                           out_features +
                                                       co];
                            }
                        }
                    }
                }
            }
        }
    }

    // 计算 grad_input
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                for (int c = 0; c < in_features; ++c) {
                    for (int co = 0; co < out_features; ++co) {
                        for (int i = 0; i < filter_size; ++i) {
                            for (int j = 0; j < filter_size; ++j) {
                                int image_h =
                                    h - pad + i; // 计算对应的输入图像位置
                                int image_w = w - pad + j;

                                if (image_h >= 0 && image_h < height &&
                                    image_w >= 0 && image_w < width) {
                                    grad_input
                                        .data()[(b * in_features * height *
                                                 width) +
                                                (c * height * width) +
                                                (image_h * width) + image_w] +=
                                        weight.data()[(co * (9 * in_features)) +
                                                      (c * (filter_size *
                                                            filter_size)) +
                                                      (i * filter_size + j)] *
                                        grad_output
                                            .data()[(b * (height * width) +
                                                     h * width + w) *
                                                        out_features +
                                                    co];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// 串行实现池化层
void forward_maxpool_serial(Tensor input, Tensor output, Tensor mask,
                            int batch_size, int in_features, int height,
                            int width) {
    int kernel_size = 2; // 2x2 池化核
    int stride = 2;      // 步幅为2

    // 计算可以池化的高度和宽度
    int output_height = height / 2;
    int output_width = width / 2;

    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < in_features; ++c) {
            for (int h = 0; h < output_height; ++h) {
                for (int w = 0; w < output_width; ++w) {
                    float max_value = -std::numeric_limits<float>::infinity();
                    int max_h = -1;
                    int max_w = -1;

                    // 在2x2窗口中找到最大值
                    for (int i = 0; i < kernel_size; ++i) {
                        for (int j = 0; j < kernel_size; ++j) {
                            int input_h = h * stride + i;
                            int input_w = w * stride + j;

                            if (input_h < height && input_w < width) {
                                float value =
                                    input.data()[(b * in_features + c) *
                                                     height * width +
                                                 input_h * width + input_w];
                                if (value > max_value) {
                                    max_value = value;
                                    max_h = input_h; // 记录最大值的高度
                                    max_w = input_w; // 记录最大值的宽度
                                }
                            }
                        }
                    }

                    // 将最大值存入输出，并在mask中记录最大值的位置
                    output.data()[(b * in_features + c) * output_height *
                                      output_width +
                                  h * output_width + w] = max_value;

                    // 在mask中标记最大值的位置
                    mask.data()[(b * in_features + c) * height * width +
                                max_h * width + max_w] = 1.0f;
                }
            }
        }
    }
}
void backward_maxpool_serial(Tensor grad_output, Tensor grad_input, Tensor mask,
                             int batch_size, int in_features, int height,
                             int width) {
    for (int b = 0; b < batch_size; ++b) {            // 遍历每个 batch
        for (int c = 0; c < in_features; ++c) {       // 遍历每个通道
            for (int h = 0; h < height / 2; ++h) {    // 遍历输出高度
                for (int w = 0; w < width / 2; ++w) { // 遍历输出宽度
                    // 计算在 grad_output 中的索引
                    int output_index =
                        b * in_features * (height / 2) * (width / 2) +
                        c * (height / 2) * (width / 2) + h * (width / 2) + w;

                    // 计算 grad_output 中的梯度
                    float grad_val = grad_output.data()[output_index];

                    // 计算输入位置，根据 mask 找到对应位置
                    for (int ph = 0; ph < 2; ++ph) { // 池化窗口的高度
                        for (int pw = 0; pw < 2; ++pw) { // 池化窗口的宽度
                            int h_in = h * 2 + ph;       // 输入高度
                            int w_in = w * 2 + pw;       // 输入宽度

                            // 检查边界
                            if (h_in < height && w_in < width) {
                                // 计算输入的索引
                                int input_index =
                                    b * in_features * height * width +
                                    c * height * width + h_in * width + w_in;

                                // 根据 mask 传播梯度
                                if (mask.data()[input_index] == 1.0f) {
                                    grad_input.data()[input_index] += grad_val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// 串行实现softmax层
void forward_softmax_serial(Tensor input, Tensor output, int batch_size,
                            int in_features) {
    for (int b = 0; b < batch_size; ++b) {
        // 计算当前行的最大值，以提高数值稳定性
        float max_val = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < in_features; ++i) {
            max_val = std::max(max_val, input.data()[b * in_features + i]);
        }

        float sum_exp = 0.0f;
        for (int i = 0; i < in_features; ++i) {
            // 计算每个元素的指数
            float exp_val =
                std::exp(input.data()[b * in_features + i] - max_val);
            output.data()[b * in_features + i] = exp_val; // 存储指数值
            sum_exp += exp_val;                           // 计算指数和
        }

        // 计算 Softmax 值
        for (int i = 0; i < in_features; ++i) {
            output.data()[b * in_features + i] /= sum_exp; // 归一化
        }
    }
}

// 串行计算cross_entropy_loss
void forward_cross_entropy_serial(Tensor input, Tensor loss, Tensor label,
                                  int batch_size, int in_features) {
    // 遍历每个样本
    for (int b = 0; b < batch_size; ++b) {
        // 获取当前样本的真实标签
        int true_label = static_cast<int>(label.data()[b]);

        // 使用安全的指数运算，确保没有溢出
        float softmax_output = input.data()[b * in_features + true_label];
        float log_prob = std::log(softmax_output);

        // 存储当前样本的损失
        loss.data()[b] = -log_prob; // 记录每一行的损失
    }
}
void backward_cross_entropy_serial(Tensor input, Tensor grad_loss, Tensor label,
                                   int batch_size, int in_features) {
    // 遍历每个样本
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < in_features; ++i) {
            if (i != int(label.data()[b]))
                grad_loss.data()[b * in_features + i] =
                    input.data()[b * in_features + i];
            else
                grad_loss.data()[b * in_features + i] =
                    input.data()[b * in_features + i] - 1.0;
        }
    }
}

int main() {
    // 全连接层测试
    std::cout << "全连接层正向传播" << std::endl;
    int batch_size = 2;
    int in_features = 3;
    int out_features = 4;

    Tensor input({batch_size, in_features}, Device::CPU);
    Tensor weight({out_features, in_features}, Device::CPU);
    Tensor output1({batch_size, out_features}, Device::GPU);
    Tensor output2({batch_size, out_features}, Device::CPU);
    input.assign({1, 2, 3, 4, 5, 6});
    std::cout << "input:" << std::endl;
    input.Print();
    input = input.gpu();
    weight.assign({0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2});
    std::cout << "weight:" << std::endl;
    weight.Print();
    weight = weight.gpu();
    forward_fc(input, output1, weight, 2, 3, 4);
    std::cout << "forward_fc算出的output:" << std::endl;
    output1.cpu().Print();
    forward_fc_serial(input.cpu(), weight.cpu(), output2);
    std::cout << "串行检验的output" << std::endl;
    output2.Print();
    std::cout << std::endl;

    std::cout << "全连接层反向传播" << std::endl;
    Tensor grad_output({batch_size, out_features}, Device::CPU);
    grad_output.assign({0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8});
    std::cout << "grad_output:" << std::endl;
    grad_output.Print();
    grad_output = grad_output.gpu();
    Tensor grad_input1({batch_size, in_features}, Device::GPU);
    Tensor grad_weight1({out_features, in_features}, Device::GPU);
    Tensor grad_input2({batch_size, in_features}, Device::CPU);
    Tensor grad_weight2({out_features, in_features}, Device::CPU);
    backward_fc(input, output1, weight, 2, 3, 4, grad_output, grad_input1,
                grad_weight1);
    std::cout << "backward_fc算出的grad_input:" << std::endl;
    grad_input1.cpu().Print();
    backward_fc_serial(input.cpu(), output1.cpu(), weight.cpu(), 2, 3, 4,
                       grad_output.cpu(), grad_input2, grad_weight2);
    std::cout << "串行检验的grad_input:" << std::endl;
    grad_input2.Print();
    std::cout << "backward_fc算出的grad_weight:" << std::endl;
    grad_weight1.cpu().Print();
    std::cout << "串行检验的grad_weight:" << std::endl;
    grad_weight2.Print();
    std::cout << "-------------------------------------------------------"
              << std::endl;

    // 卷积层测试
    std::cout << "卷积层正向传播" << std::endl;
    batch_size = 2;
    in_features = 3;
    out_features = 2;
    int height = 4;
    int width = 4;

    Tensor input_c({batch_size, in_features, height, width}, Device::CPU);
    Tensor weight_c({out_features, in_features, 3, 3}, Device::CPU);
    Tensor output_c1({batch_size, out_features, height, width}, Device::GPU);
    Tensor output_c2({batch_size, out_features, height, width}, Device::CPU);
    Tensor column({batch_size, width * height, 9 * in_features}, Device::GPU);

    input_c.assign(
        {2.7, 3.2, 1.8, 4.8, 4.8, 0.4, 1.8, 3.4, 3.0, 2.3, 5.0, 2.5, 3.6, 3.5,
         1.2, 1.0, 0.8, 4.3, 0.2, 4.6, 3.5, 0.4, 0.2, 3.9, 3.8, 4.8, 2.3, 2.7,
         3.4, 0.8, 2.5, 4.4, 3.3, 1.0, 1.8, 3.3, 4.2, 0.1, 4.0, 0.9, 2.7, 1.8,
         2.0, 1.6, 2.7, 4.5, 1.5, 1.0, 2.0, 3.7, 0.9, 1.4, 1.2, 2.0, 2.3, 1.1,
         4.9, 0.6, 3.0, 0.9, 2.7, 4.9, 2.2, 4.6, 3.9, 2.8, 5.0, 4.7, 3.7, 0.1,
         1.5, 0.1, 0.1, 3.1, 5.0, 2.6, 3.9, 3.7, 1.1, 3.5, 3.0, 3.6, 1.8, 0.4,
         3.0, 2.1, 0.7, 2.4, 1.8, 0.7, 3.5, 2.5, 3.2, 3.7, 3.7, 0.7});
    input_c = input_c.gpu();

    weight_c.assign({0.7,  -0.8, 0.6, 0.1,  -0.3, 1.0,  -1.0, -0.2, 0.5,
                     0.3,  0.5,  0.6, 0.8,  -0.7, -0.8, -0.3, 0.7,  0.2,
                     0.6,  -0.4, 0.4, 0.0,  0.2,  -0.4, -0.7, 0.9,  0.0,
                     0.4,  -0.6, 0.5, -0.6, -0.2, -0.5, -0.5, 0.7,  1.0,
                     0.5,  0.2,  0.7, 0.8,  -0.1, -0.2, -0.7, 0.4,  0.2,
                     -0.4, -0.5, 0.1, 0.4,  0.7,  0.4,  0.5,  -0.6, -0.2});
    weight_c = weight_c.gpu();

    forward_conv(input_c, column, weight_c, output_c1, batch_size, in_features,
                 out_features, height, width);
    std::cout << "forward_conv算出的output" << std::endl;
    output_c1.cpu().Print();

    forward_conv_serial(input_c.cpu(), weight_c.cpu(), output_c2, batch_size,
                        in_features, out_features, height, width);
    std::cout << "串行检验的output" << std::endl;
    output_c2.Print();
    std::cout << std::endl;

    std::cout << "卷积层反向传播" << std::endl;
    Tensor grad_output_c({batch_size, out_features, height, width},
                         Device::CPU);
    grad_output_c.assign({2.5, 2.3, 1.3, 1.4, 1.4, 1.4, 1.2, 2.2, 1.2, 2.6, 3.0,
                          2.7, 1.9, 2.3, 2.3, 2.3, 1.7, 1.0, 2.3, 2.6, 2.0, 1.7,
                          1.6, 1.3, 2.6, 2.1, 2.4, 2.8, 1.6, 2.6, 2.0, 1.5, 1.4,
                          3.0, 2.5, 2.4, 1.7, 1.6, 1.3, 2.3, 2.4, 1.8, 1.3, 2.6,
                          2.6, 1.4, 2.5, 2.7, 2.7, 1.2, 1.1, 1.0, 2.8, 1.9, 1.2,
                          1.4, 1.9, 1.3, 2.5, 1.2, 2.3, 1.7, 2.9, 2.1});
    grad_output_c = grad_output_c.gpu();
    Tensor grad_input_c1({batch_size, in_features, height, width}, Device::GPU);
    Tensor grad_input_c2({batch_size, in_features, height, width}, Device::CPU);
    Tensor grad_weight_c1({out_features, in_features, 3, 3}, Device::GPU);
    Tensor grad_weight_c2({out_features, in_features, 3, 3}, Device::CPU);

    Tensor grad_column({batch_size, width * height, 9 * in_features},
                       Device::GPU);
    backward_conv(column, grad_column, weight_c, batch_size, in_features,
                  out_features, height, width, grad_output_c, grad_input_c1,
                  grad_weight_c1);
    std::cout << "backward_conv算出的grad_input:" << std::endl;
    grad_input_c1.cpu().Print();
    backward_conv_serial(column.cpu(), weight_c.cpu(), batch_size, in_features,
                         out_features, height, width, grad_output_c.cpu(),
                         grad_input_c2, grad_weight_c2);
    std::cout << "串行检验的grad_input:" << std::endl;
    grad_input_c2.Print();
    std::cout << "backward_conv算出的grad_weight:" << std::endl;
    grad_weight_c1.cpu().Print();
    std::cout << "串行检验的grad_weight:" << std::endl;
    grad_weight_c2.Print();
    std::cout << "-------------------------------------------------------"
              << std::endl;

    // 池化层测试
    std::cout << "池化层正向传播" << std::endl;
    batch_size = 2;
    in_features = 3;
    height = 5;
    width = 6;
    Tensor input_p({batch_size, in_features, height, width}, Device::CPU);
    input_p.assign(
        {2.7,  4.2,  2.1,  2.4,  -0.6, 3.1,  4.0,  3.4,  -0.1, 3.1,  3.8, 1.7,
         -0.6, 2.4,  3.4,  4.2,  1.0,  1.4,  2.3,  4.5,  0.6,  4.6,  3.9, 4.9,
         3.7,  0.5,  -0.9, 5.0,  -0.6, 3.8,  3.4,  -0.5, 4.1,  2.0,  2.5, 2.9,
         4.7,  -0.2, 2.0,  0.8,  1.0,  1.4,  2.7,  2.3,  4.9,  1.2,  0.4, -0.1,
         4.3,  3.0,  4.7,  1.5,  0.2,  1.1,  -0.8, 2.5,  -0.1, -0.1, 3.3, 2.0,
         2.1,  3.8,  4.7,  4.4,  3.8,  4.4,  4.1,  3.1,  4.5,  4.1,  0.6, -0.2,
         0.8,  3.9,  2.3,  4.1,  4.1,  2.6,  0.1,  -0.0, 3.5,  4.0,  4.9, 2.7,
         -0.6, 1.9,  -0.2, 4.4,  0.7,  3.0,  2.8,  4.4,  1.1,  5.0,  1.2, 0.8,
         3.3,  2.3,  0.8,  3.7,  -0.9, 1.5,  4.3,  3.8,  -0.2, 2.1,  0.7, 0.1,
         0.1,  2.5,  2.7,  -0.2, 2.3,  -0.3, 0.3,  0.5,  1.4,  0.5,  2.2, -0.8,
         0.5,  4.8,  4.2,  -0.0, 0.1,  4.7,  3.6,  3.8,  1.7,  2.5,  1.2, -0.7,
         3.9,  4.1,  2.1,  1.7,  3.3,  -0.7, 0.3,  2.6,  -0.8, 3.1,  0.2, 3.0,
         1.8,  3.1,  0.3,  4.8,  -0.9, 4.8,  3.9,  4.8,  1.9,  1.6,  3.2, 4.6,
         2.4,  3.6,  1.8,  0.4,  4.6,  0.8,  3.6,  1.3,  3.8,  -0.5, 3.8, 4.2,
         4.3,  1.5,  0.8,  1.7,  -0.4, 3.2,  3.2,  0.6,  4.6,  4.9,  1.7, 0.9});
    input_p = input_p.gpu();
    Tensor output_p1({batch_size, in_features, height / 2, width / 2},
                     Device::GPU);
    Tensor output_p2({batch_size, in_features, height / 2, width / 2},
                     Device::CPU);
    Tensor mask1({batch_size, in_features, height, width}, Device::GPU);
    Tensor mask2({batch_size, in_features, height, width}, Device::CPU);
    forward_maxpool(input_p, output_p1, mask1, batch_size, in_features, height,
                    width);
    forward_maxpool_serial(input_p.cpu(), output_p2, mask2, batch_size,
                           in_features, height, width);
    std::cout << "forward_maxpool算出的output:" << std::endl;
    output_p1.cpu().Print();
    std::cout << "串行检验的output:" << std::endl;
    output_p2.Print();
    std::cout << "forward_maxpool算出的mask:" << std::endl;
    mask1.cpu().Print();
    std::cout << "串行检验的mask:" << std::endl;
    mask2.Print();
    std::cout << std::endl;

    std::cout << "池化层反向传播" << std::endl;
    Tensor grad_output_p({batch_size, in_features, height / 2, width / 2},
                         Device::CPU);
    grad_output_p.assign({0.5, 1.1, 1.7, 1.9, 0.4, 0.7, 1.1, 1.4, 0.1,
                          1.1, 1.8, 0.6, 1.4, 0.1, 0.7, 0.1, 0.2, 1.2,
                          0.1, 0.3, 1.1, 1.9, 1.8, 1.6, 1.0, 0.7, 1.2,
                          0.8, 0.5, 0.0, 1.6, 0.5, 0.5, 1.0, 1.3, 0.4});
    grad_output_p = grad_output_p.gpu();
    Tensor grad_input_p1({batch_size, in_features, height, width}, Device::GPU);
    Tensor grad_input_p2({batch_size, in_features, height, width}, Device::CPU);
    backward_maxpool(grad_output_p, grad_input_p1, mask1, batch_size,
                     in_features, height, width);
    backward_maxpool_serial(grad_output_p.cpu(), grad_input_p2, mask2,
                            batch_size, in_features, height, width);

    std::cout << "backward_maxpool算出的grad_input:" << std::endl;
    grad_input_p1.cpu().Print();
    std::cout << "串行检验的grad_input:" << std::endl;
    grad_input_p2.Print();
    std::cout << "-------------------------------------------------------"
              << std::endl;

    // softmax层测试
    std::cout << "Softmax层正向传播" << std::endl;
    batch_size = 5;
    in_features = 10;
    Tensor input_s({batch_size, in_features}, Device::CPU);
    input_s.assign({3.8,  2.3,  2.2, -0.3, 3.0,  -0.4, 3.3, 4.8, 0.1,  0.8,
                    4.2,  3.5,  2.4, 3.9,  -0.2, 1.7,  4.8, 0.2, -1.3, 0.7,
                    -0.0, 4.1,  0.6, -0.9, 3.2,  0.4,  0.2, 2.2, 0.7,  -1.3,
                    -0.5, 4.1,  1.2, 2.9,  2.4,  3.5,  2.1, 2.0, -0.3, -0.7,
                    -0.4, -1.0, 0.6, 3.2,  -1.7, 1.0,  4.6, 1.2, 1.5,  -2.0});
    input_s = input_s.gpu();
    Tensor output_s1({batch_size, in_features}, Device::GPU);
    Tensor output_s2({batch_size, in_features}, Device::CPU);
    forward_softmax(input_s, output_s1, batch_size, in_features);
    forward_softmax_serial(input_s.cpu(), output_s2, batch_size, in_features);
    std::cout << "forward_softmax算出的output:" << std::endl;
    output_s1.cpu().Print();
    std::cout << "串行检验的output:" << std::endl;
    output_s2.Print();
    std::cout << "-------------------------------------------------------"
              << std::endl;

    // cross_entropy_loss 测试
    std::cout << "CrossEntropyLoss 正向传播" << std::endl;
    Tensor label({batch_size}, Device::CPU);
    label.assign({1, 3, 5, 7, 9});
    label = label.gpu();
    Tensor loss1({batch_size}, Device::GPU);
    Tensor loss2({batch_size}, Device::CPU);
    forward_cross_entropy(output_s1, loss1, label, batch_size, in_features);
    forward_cross_entropy_serial(output_s2, loss2, label.cpu(), batch_size,
                                 in_features);
    std::cout << "forward_cross_entropy算出的loss:" << std::endl;
    loss1.cpu().Print();
    std::cout << "串行检验的loss:" << std::endl;
    loss2.Print();
    std::cout << std::endl;

    std::cout << "CrossEntropyLoss 反向传播" << std::endl;
    Tensor grad_loss1({batch_size, in_features}, Device::GPU);
    Tensor grad_loss2({batch_size, in_features}, Device::CPU);
    backward_cross_entropy(output_s1, grad_loss1, label, batch_size,
                           in_features);
    backward_cross_entropy_serial(output_s2, grad_loss2, label.cpu(),
                                  batch_size, in_features);
    std::cout << "backward_cross_entropy算出的grad_output:" << std::endl;
    grad_loss1.cpu().Print();
    std::cout << "串行检验的grad_output:" << std::endl;
    grad_loss2.Print();
    std::cout << "-------------------------------------------------------"
              << std::endl;
    return 0;
}