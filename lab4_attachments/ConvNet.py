import numpy as np
import MyTensor as mt

class ConvNet:
    def __init__(self, input_shape):
        # 假设输入是 (batch_size, channels, height, width)
        self.input_shape = input_shape
        self.layer1_weight = mt.Tensor([16, input_shape[1], 3, 3], device=mt.Device.GPU)  # 卷积层1的权重
        self.layer2_weight = mt.Tensor([32, 16, 3, 3], device=mt.Device.GPU)  # 卷积层2的权重
        self.fc1_weight = mt.Tensor([512, 32 * 7 * 7], device=mt.Device.GPU)  # 全连接层1的权重
        self.fc2_weight = mt.Tensor([10, 512], device=mt.Device.GPU)  # 全连接层2的权重
        
        # 初始化权重
        self.layer1_weight.assign(np.random.randn(*self.layer1_weight.shape).astype(np.float32))
        self.layer2_weight.assign(np.random.randn(*self.layer2_weight.shape).astype(np.float32))
        self.fc1_weight.assign(np.random.randn(*self.fc1_weight.shape).astype(np.float32))
        self.fc2_weight.assign(np.random.randn(*self.fc2_weight.shape).astype(np.float32))

    def forward(self, input_tensor):
        # forward pass
        self.conv1_output = mt.Tensor([input_tensor.shape[0], 16, 28, 28], device=mt.Device.GPU)
        mt.forward_conv(input_tensor, self.layer1_weight, self.conv1_output)  # 卷积层1

        self.relu1_output = mt.Tensor(self.conv1_output.shape, device=mt.Device.GPU)
        mt.forward_relu(self.conv1_output, self.relu1_output)  # ReLU激活

        self.conv2_output = mt.Tensor([input_tensor.shape[0], 32, 14, 14], device=mt.Device.GPU)
        mt.forward_conv(self.relu1_output, self.layer2_weight, self.conv2_output)  # 卷积层2

        self.relu2_output = mt.Tensor(self.conv2_output.shape, device=mt.Device.GPU)
        mt.forward_relu(self.conv2_output, self.relu2_output)  # ReLU激活

        self.pool2_output = mt.Tensor([input_tensor.shape[0], 32, 7, 7], device=mt.Device.GPU)
        mt.forward_maxpool(self.relu2_output, self.pool2_output)  # 最大池化层2

        self.flattened = mt.Tensor([input_tensor.shape[0], 32 * 7 * 7], device=mt.Device.GPU)
        mt.flatten(self.pool2_output, self.flattened)  # 展平操作

        self.fc1_output = mt.Tensor([input_tensor.shape[0], 512], device=mt.Device.GPU)
        mt.forward_fc(self.flattened, self.fc1_weight, self.fc1_output)  # 全连接层1

        self.fc2_output = mt.Tensor([input_tensor.shape[0], 10], device=mt.Device.GPU)
        mt.forward_fc(self.fc1_output, self.fc2_weight, self.fc2_output)  # 全连接层2
        
        return self.fc2_output

    def backward(self, loss_tensor):
        # backward pass
        loss_tensor.backward()  # 计算图的反向传播

        # 按顺序反向传播
        mt.backward_fc(self.fc2_output, self.fc1_output, self.fc2_weight, loss_tensor.grad)  # 全连接层2
        mt.backward_fc(self.fc1_output, self.flattened, self.fc1_weight, self.fc1_output.grad)  # 全连接层1

        mt.backward_maxpool(self.pool2_output, self.relu2_output, self.pool2_output.grad)  # 池化层2
        mt.backward_conv(self.conv2_output, self.relu2_output, self.layer2_weight, self.pool2_output.grad)  # 卷积层2

        mt.backward_maxpool(self.pool2_output, self.relu1_output, self.pool2_output.grad)  # 池化层1
        mt.backward_conv(self.conv1_output, self.relu1_output, self.layer1_weight, self.pool2_output.grad)  # 卷积层1
