import torchvision
from torchvision import transforms
import numpy as np
import MyTensor

# 加载 MNIST 数据集
transform = transforms.Compose([
    transforms.ToTensor()  # 将图像转为 [0, 1] 范围内的张量
])

train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)

test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform)



