import numpy as np
from operators import *
from tensor import TensorFull

class ConvNet:
    def __init__(self):
        # 初始化卷积神经网络各层
        self.conv_weight = TensorFull(np.random.randn(32, 1, 3, 3).astype(np.float32) / np.sqrt(32), requires_grad=True)
        self.fc1_weight = TensorFull(np.random.randn(128, 14 * 14 * 32).astype(np.float32) / np.sqrt(14 * 14 * 32), requires_grad=True)
        self.fc2_weight = TensorFull(np.random.randn(10, 128).astype(np.float32) / np.sqrt(128), requires_grad=True)
        

    def forward(self, x):
        # 卷积层
        x = conv(x, self.conv_weight)
        # print("卷积1")
        # print(np.sum(x.numpy()))
        x = maxpool(x)
        # print("池化1")
        # print(np.sum(x.numpy()))

        x = relu(x)
        # print("Relu 1")
        # print(np.sum(x.numpy()))

        # Flatten
        x = flatten(x)
        # print("扁平")
        # print(np.sum(x.numpy()))

        # 全连接层1
        x = linear(x, self.fc1_weight)
        # print("全连接1")
        # print(np.sum(x.numpy()))

        x = relu(x)
        # print("Relu 2")
        # print(np.sum(x.numpy()))

        # 全连接层2
        x = linear(x, self.fc2_weight)
        # print("全连接2")
        # print(np.sum(x.numpy()))
        return x

    def backward(self, loss):
        # 计算梯度
        loss.backward()

    def update_parameters(self, lr):
        # Adam or other optimizer can be applied here, but we are simplifying to basic SGD for now
        for param in [self.conv_weight, self.fc1_weight, self.fc2_weight]:
            param -= lr * param.grad  # Simple gradient descent step

    def train(self, train_images_tensor, train_labels_tensor, epochs, lr):
        total_samples = len(train_images_tensor)
        for epoch in range(epochs):
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
                print(np.sum(loss.numpy()))
                total_loss += np.sum(loss.numpy())

                # Backward pass
                self.backward(loss)

                # 参数更新
                self.update_parameters(lr)

            print(f"Epoch {epoch+1}, Loss: {total_loss}")

    def predict(self, test_images_tensor, test_labels_tensor):
        correct_predictions = 0
        total_samples = len(test_images_tensor)

        for i in range(total_samples):
            # 获取当前测试图像的数据和标签
            data = test_images_tensor[i]  # 取一张图片
            target = test_labels_tensor[i]  # 取对应的标签

            # Forward pass
            output = self.forward(data)

            # 获取预测的类别
            predicted_class = np.argmax(output.numpy(), axis=-1)
            
            # 判断预测是否正确
            if predicted_class == np.argmax(target.numpy(), axis=-1):
                correct_predictions += 1

        # 计算准确率
        accuracy = correct_predictions / total_samples
        print(f"Accuracy: {accuracy * 100:.2f}%")
        return accuracy