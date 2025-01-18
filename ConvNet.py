import numpy as np
from operators import *
from tensor import TensorFull

class ConvNet:
    def __init__(self):
        # 初始化卷积神经网络各层
        self.conv1_weight = TensorFull(np.random.randn(32, 1, 5, 5).astype(np.float32), requires_grad=True)
        self.conv2_weight = TensorFull(np.random.randn(64, 32, 5, 5).astype(np.float32), requires_grad=True)
        self.fc1_weight = TensorFull(np.random.randn(7 * 7 * 64, 128).astype(np.float32), requires_grad=True)
        self.fc2_weight = TensorFull(np.random.randn(128, 10).astype(np.float32), requires_grad=True)
        

    def forward(self, x):
        # 卷积层1
        x = conv(x, self.conv1_weight)
        x = maxpool(x)

        # 卷积层2
        x = conv(x, self.conv2_weight)
        x = maxpool(x)

        # Flatten
        x = reshape((x.shape()[0], -1))

        # 全连接层1
        x = linear(x, self.fc1_weight)

        # 全连接层2
        x = linear(x, self.fc2_weight)

        return x

    def backward(self, loss):
        # 计算梯度
        loss.backward()

    def update_parameters(self, lr):
        # Adam or other optimizer can be applied here, but we are simplifying to basic SGD for now
        for param in [self.conv1_weight, self.conv2_weight, self.fc1_weight, self.fc2_weight]:
            param -= lr * param.grad  # Simple gradient descent step

    def train(self, train_loader, epochs, lr):
        for epoch in range(epochs):
            total_loss = 0
            for data, target in train_loader:
                # Forward pass
                output = self.forward(data)

                # 计算损失
                loss = cross_entropy(output, target)
                total_loss += np.sum(loss.numpy())
                # Backward pass
                self.backward(loss)

                # 参数更新
                self.update_parameters(lr)

            print(f"Epoch {epoch+1}, Loss: {total_loss}")