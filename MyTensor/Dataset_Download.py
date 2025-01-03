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

# 转换为 NumPy 数组
train_images = train_dataset.data.numpy().astype(np.float32)
train_labels = train_dataset.targets.numpy()

# 转换为tensor
batch_size = 64
batch_images = train_images[:batch_size]  # 获取前 64 张图片
batch_tensors = [MyTensor.tensor_from_numpy(image) for image in batch_images]

# 查看结果
for i, tensor in enumerate(batch_tensors[:5]):  # 打印前 5 个 Tensor
    print(f"Tensor {i}:")
    tensor.Print()
