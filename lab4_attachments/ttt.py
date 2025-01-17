import numpy as np
from task1_operators import *

# 定义一个简单的加法运算
def test_addition():
    # 创建两个Tensor，要求计算梯度
    a = Tensor(np.random.randn(2, 3), requires_grad=True)
    b = Tensor(np.random.randn(2, 3), requires_grad=True)
    
    # 进行加法运算
    result = a + b
    
    # 执行反向传播计算梯度
    result.backward()

    # 打印梯度
    print(f"a.grad:\n{a.grad.numpy()}")
    print(f"b.grad:\n{b.grad.numpy()}")

# 定义一个简单的乘法运算
def test_multiplication():
    # 创建两个Tensor，要求计算梯度
    a = Tensor(np.random.randn(2, 3), requires_grad=True)
    b = Tensor(np.random.randn(2, 3), requires_grad=True)
    
    # 进行乘法运算
    result = a * b
    
    # 执行反向传播计算梯度
    result.backward()

    # 打印梯度
    print(f"a.grad:\n{a.grad.numpy()}")
    print(f"b.grad:\n{b.grad.numpy()}")

# 定义一个标量加法运算
def test_add_scalar():
    # 创建一个Tensor，要求计算梯度
    a = Tensor(np.random.randn(2, 3), requires_grad=True)
    
    # 进行加法运算
    result = a + 5.0
    
    # 执行反向传播计算梯度
    result.backward()

    # 打印梯度
    print(f"a.grad:\n{a.grad.numpy()}")

# 定义一个指数运算
def test_exponentiation():
    # 创建一个Tensor，要求计算梯度
    a = Tensor(np.random.randn(2, 3), requires_grad=True)
    
    # 进行指数运算
    result = a ** 2
    
    # 执行反向传播计算梯度
    result.backward()

    # 打印梯度
    print(f"a.grad:\n{a.grad.numpy()}")

# 定义一个ReLU运算
def test_relu():
    # 创建一个Tensor，要求计算梯度
    a = Tensor(np.random.randn(2, 3), requires_grad=True)
    
    # 进行ReLU运算
    result = relu(a)
    
    # 执行反向传播计算梯度
    result.backward()

    # 打印梯度
    print(f"a.grad:\n{a.grad.numpy()}")

# 定义一个Log运算
def test_log():
    # 创建一个Tensor，要求计算梯度
    a = Tensor(np.random.randn(2, 3) + 2.0, requires_grad=True)  # 防止负数输入
    result = log(a)
    
    # 执行反向传播计算梯度
    result.backward()

    # 打印梯度
    print(f"a.grad:\n{a.grad.numpy()}")

# 运行所有测试
def run_gradient_tests():
    print("Testing Addition:")
    test_addition()
    print("\nTesting Multiplication:")
    test_multiplication()
    print("\nTesting Add Scalar:")
    test_add_scalar()
    print("\nTesting Exponentiation:")
    test_exponentiation()
    print("\nTesting ReLU:")
    test_relu()
    print("\nTesting Log:")
    test_log()

# 执行测试
run_gradient_tests()
