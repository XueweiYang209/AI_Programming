from ConvNet import *
from dataset_download import *

if __name__ == '__main__':
    # 转换为 NumPy 数组
    train_images = train_dataset.data.numpy().astype(np.float32)
    train_labels = train_dataset.targets.numpy()

    # 转换为 TensorFull 格式
    train_images_tensor = [TensorFull(image, requires_grad=False) for image in train_images]
    train_labels_tensor = [TensorFull(label, requires_grad=False) for label in train_labels]

    # 实例化模型
    model = ConvNet()

    # 训练配置
    epochs = 5  # 训练 5 个 epoch
    learning_rate = 0.001

    # 开始训练
    model.train(train_images_tensor, train_labels_tensor, epochs, learning_rate)

    # 进行预测
    test_images = test_dataset.data.numpy().astype(np.float32)
    test_labels = test_dataset.targets.numpy()
    test_images_tensor = [TensorFull(image, requires_grad=False) for image in test_images]
    test_labels_tensor = [TensorFull(label, requires_grad=False) for label in test_labels]
    accuracy = model.predict(test_images_tensor, test_labels_tensor)
    print(accuracy)