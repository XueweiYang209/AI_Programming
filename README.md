# Programming in AI
This is the final assignment for a course Programming in AI of Peking University, in which a simplified PyTorch-alike AI framework is established to realize CNN model.

# 作者
Xuewei Yang

# 文件主要内容

* MyTenor文件夹
    * `Tensor.h` 与 `Tensor.cu`：用于定义Tensor类，并实现一些成员变量和成员函数
    * `Module.h` 与 `Module.cu`：通过CUDA kernel实现Tensor的卷积神经网络算子和张量算子的前向传播与反向传播
    * `CMakeLists.txt`：CMake文件，编译生成动态链接库MyTensor
    * `binding.cpp`：用于绑定相关类与函数到Python中
    * `UnitTest.py`：基于绑定库MyTensor的测试文件，通过与PyTorch相关api的结果对比，验证MyTensor实现的正确性

* `basic_operator.py`：实现基本运算符类Op和计算图上的节点类Value
* `operators.py`：实现了Value的继承类Tensor，基于Op的继承类TensorOp实现了各种具体的运算符类，包括张量算子和基于MyTensor运算的卷积神经网络算子
* `tensor.py`：继承自Tensor类实现了TensorFull类
* `autodiff.py`：实现拓扑排序与自动微分
* `dataset_download.py`：实现了一个DataLoader类，用于分批次获取数据。另外通过Torch加载并处理了MINST数据集
* `ConvNet.py`：实现了一个ConvNet类，可以初始化卷积神经网络模型，实现模型的前向传播、反向传播、参数优化、训练和预测等方法
* `main.py`：实例化模型ConvNet，训练和预测
* `test_forward.py` 和 `test_topo_sort.py`：测试文件。可用于测试Tensor的前向传播和拓扑排序等功能
* `test_backward.py`：测试文件。简单生成一些测试用例用于观察Tensor反向传播的功能

# 验证运行
在`ConvNet.py`中的forward函数自定义卷积神经网络结构，在`main.py`中调整相应超参数，直接运行即可训练并预测
