import numpy as np
from operators import *
from tensor import TensorFull
from dataset_download import DataLoader

class ConvNet:
    def __init__(self):
        # 初始化卷积神经网络各层
        self.conv_weight = TensorFull(np.random.randn(32, 1, 3, 3).astype(np.float32) / np.sqrt(32), requires_grad=True)
        self.fc1_weight = TensorFull(np.random.randn(128, 14 * 14 * 32).astype(np.float32) / np.sqrt(14 * 14 * 32), requires_grad=True)
        self.fc2_weight = TensorFull(np.random.randn(10, 128).astype(np.float32) / np.sqrt(128), requires_grad=True)
        

    def forward(self, x):
        # 卷积层
        x = conv(x, self.conv_weight)
        x = maxpool(x)

        x = relu(x)

        # Flatten
        x = flatten(x)

        # 全连接层1
        x = linear(x, self.fc1_weight)

        x = relu(x)

        # 全连接层2
        x = linear(x, self.fc2_weight)
        return x

    def backward(self, loss):
        # 计算梯度
        loss.backward()

    def update_parameters(self, lr):
        # SGD算法
        for param in [self.conv_weight, self.fc1_weight, self.fc2_weight]:
            param.data -= lr * param.grad.data

    def train(self, train_images, train_labels, epochs, lr, batch_size):
        data_loader = DataLoader(train_images, train_labels, batch_size)
        
        for epoch in range(1):
            for i, (batch_images, batch_labels) in enumerate(data_loader):
                if i > 119:
                    break
                # 批量数据训练
                # Forward pass
                output = self.forward(batch_images)

                # 计算损失
                loss = cross_entropy(output, batch_labels)
                print(f"第{i+1}次循环")
                print(f"loss: {np.mean(loss.numpy())}")

                # Backward pass
                self.backward(loss)

                # 参数更新
                self.update_parameters(lr)

    def predict(self, test_images, test_labels, batch_size=1):
        correct_predictions = 0
        total_samples = test_images.shape[0]

        # 创建数据加载器
        test_loader = DataLoader(test_images, test_labels, batch_size)

        for i, (batch_data, batch_labels) in enumerate(test_loader):
            # Forward pass
            output = self.forward(batch_data)

            # 获取预测的类别
            predicted_class = np.argmax(output.numpy(), axis=-1)
            
            # 判断预测是否正确
            correct_predictions += np.sum(predicted_class == np.argmax(batch_labels.numpy(), axis=-1))

            # 打印当前准确率
            print("current_accuracy:", correct_predictions / (i + 1))

        # 计算总准确率
        accuracy = correct_predictions / total_samples
        print(f"Accuracy: {accuracy * 100:.2f}%")
        return accuracy