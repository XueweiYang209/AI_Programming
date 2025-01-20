from ConvNet import *
from dataset_download import *

if __name__ == '__main__':
    # 转换为 NumPy 数组
    train_images = train_dataset.data.numpy().astype(np.float32)
    train_labels = train_dataset.targets.numpy()

    # 实例化模型
    model = ConvNet()

    # 训练配置
    epochs = 5  # 训练 5 个 epoch
    learning_rate = 0.001
    batch_size = 1  # 设置 batch_size

    # 开始训练
    model.train(train_images, train_labels, epochs, learning_rate, batch_size)

    # 进行预测
    test_images = test_dataset.data.numpy().astype(np.float32)
    test_labels = test_dataset.targets.numpy()
    accuracy = model.predict(test_images, test_labels)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
