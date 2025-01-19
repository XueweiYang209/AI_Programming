import numpy as np
from tensor import TensorFull
from operators import *

class SimpleNet:
    def __init__(self):
        # 初始化全连接层权重
        self.fc1_weight = TensorFull(np.random.randn(128, 28*28).astype(np.float32) / np.sqrt(28*28), requires_grad=True)
        self.fc2_weight = TensorFull(np.random.randn(10, 128).astype(np.float32) / np.sqrt(128), requires_grad=True)
        
    def forward(self, x):
        # Flatten 输入
        x = flatten(x)
        # print("扁平")
        # print(np.max(x.numpy()),np.min(x.numpy()),np.sum(x.numpy()))
        # 全连接层1
        x = linear(x, self.fc1_weight)
        # print("全连接1")
        # print(np.max(x.numpy()),np.min(x.numpy()),np.sum(x.numpy()))

        x = relu(x)
        # print("relu1")
        # print(np.max(x.numpy()),np.min(x.numpy()),np.sum(x.numpy()))

        # 全连接层2
        x = linear(x, self.fc2_weight)
        # print("全连接2")
        # print(np.max(x.numpy()),np.min(x.numpy()),np.sum(x.numpy()))

        return x

    def backward(self, loss):
        # 计算梯度
        loss.backward()

    def update_parameters(self, lr):
        # 更新参数
        for param in [self.fc1_weight, self.fc2_weight]:
            param.data -= lr * param.grad.data  # Simple gradient descent step

    def train(self, train_images_tensor, train_labels_tensor, epochs, lr):
        total_samples = len(train_images_tensor)
        for epoch in range(1):
            total_loss = 0
            for i in range(total_samples):
                # 获取当前图片的数据和标签
                data = train_images_tensor[i]  # 取一张图片
                target = train_labels_tensor[i]  # 取对应的标签

                # Forward pass
                output = self.forward(data)

                # 计算损失
                loss = cross_entropy(output, target)
                print(i)
                print(loss.numpy())
                total_loss += np.sum(loss.numpy())

                # Backward pass
                self.backward(loss)

                # 参数更新
                self.update_parameters(lr)

            print(f"Epoch {epoch+1}, Loss: {total_loss}")

