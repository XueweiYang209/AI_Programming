import torchvision
from torchvision import transforms
import numpy as np
from tensor import *

class DataLoader:
    def __init__(self, images, labels, batch_size):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.images):
            self.index = 0
            raise StopIteration
        
        batch_images = self.images[self.index : self.index + self.batch_size]
        batch_labels = self.labels[self.index : self.index + self.batch_size]

        # 将整个批次的图像和标签数据转换为 numpy 数组，再转为 TensorFull 类型
        batch_images = np.array(batch_images)  # 转为 numpy 数组
        batch_labels = np.array(batch_labels)  # 转为 numpy 数组

        batch_images_tensor = TensorFull(batch_images, requires_grad=False)
        batch_labels_tensor = TensorFull(batch_labels, requires_grad=False)

        self.index += self.batch_size
        return batch_images_tensor, batch_labels_tensor



# 加载 MNIST 数据集
transform = transforms.Compose([
    transforms.ToTensor()  # 将图像转为 [0, 1] 范围内的张量
])

train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)

test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform)



