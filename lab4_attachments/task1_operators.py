"""
此次作业借鉴和参考了Needle项目 https://github.com/dlsyscourse/lecture5
本文件我们给出一个基本完善的Tensor类，但是缺少梯度计算的功能
你需要把梯度计算所需要的运算的正反向计算补充完整
一共有12*2处
当你填写好之后，可以调用test_task1_*****.py中的函数进行测试
"""

import numpy as np
from typing import List, Optional, Tuple, Union
from device import cpu, Device
from basic_operator import Op, Value
from task2_autodiff import compute_gradient_of_variables
import MyTensor as mt

def constant(*shape, c=1.0, device=None, dtype="float32", requires_grad=False):
    """Generate constant Tensor"""
    device = cpu() if device is None else device
    array = device.ones(*shape, dtype=dtype) * c  # note: can change dtype
    return Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)
def ones(shape, device=None, dtype="float32", requires_grad=False):
    """Generate all-ones Tensor"""
    return constant(
        *shape, c=1.0, device=device, dtype=dtype, requires_grad=requires_grad
    )

class Tensor(Value):
    grad: "Tensor"

    def __init__(
        self,
        array,
        *,
        device: Optional[Device] = None,
        dtype=None,
        requires_grad=True,
        **kwargs
    ):
        if isinstance(array, Tensor):
            if device is None:
                device = array.device
            if dtype is None:
                dtype = array.dtype
            if device == array.device and dtype == array.dtype:
                cached_data = array.realize_cached_data()
            else:
                cached_data = Tensor._array_from_numpy(
                    array.numpy(), device=device, dtype=dtype
                )
        else:
            device = device if device else cpu()
            cached_data = Tensor._array_from_numpy(array, device=device, dtype=dtype)

        self._init(
            None,
            [],
            cached_data=cached_data,
            requires_grad=requires_grad,
        )

    @staticmethod
    def _array_from_numpy(numpy_array, device, dtype):
        return np.array(numpy_array, dtype=dtype)

    @staticmethod
    def make_from_op(op: Op, inputs: List["Value"]):
        tensor = Tensor.__new__(Tensor)
        tensor._init(op, inputs)
        if not tensor.requires_grad:
            return tensor.detach()
        tensor.realize_cached_data()
        return tensor

    @staticmethod
    def make_const(data, requires_grad=False):
        tensor = Tensor.__new__(Tensor)
        tensor._init(
            None,
            [],
            cached_data=data
            if not isinstance(data, Tensor)
            else data.realize_cached_data(),
            requires_grad=requires_grad,
        )
        return tensor

    @property
    def data(self):
        return self.detach()

    @data.setter
    def data(self, value):
        assert isinstance(value, Tensor)
        assert value.dtype == self.dtype, "%s %s" % (
            value.dtype,
            self.dtype,
        )
        self.cached_data = value.realize_cached_data()

    def detach(self):
        return Tensor.make_const(self.realize_cached_data())

    @property
    def shape(self):
        return self.realize_cached_data().shape

    @property
    def dtype(self):
        return self.realize_cached_data().dtype

    @property
    def device(self):
        return cpu()


    def backward(self, out_grad=None):
        if out_grad is None:
            out_grad = ones(self.shape)
        return compute_gradient_of_variables(self, out_grad)
        

    def __repr__(self):
        return "Tensor(" + str(self.realize_cached_data()) + ")"

    def __str__(self):
        return self.realize_cached_data().__str__()

    def numpy(self):
        data = self.realize_cached_data()

        return data


    def __add__(self, other):
        if isinstance(other, Tensor):
            return EWiseAdd()(self, other)
        else:
            return AddScalar(other)(self)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return EWiseMul()(self, other)
        else:
            return MulScalar(other)(self)

    def __pow__(self, other):
        if isinstance(other, Tensor):
            return EWisePow()(self, other)
        else:
            return PowerScalar(other)(self)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return EWiseAdd()(self, Negate()(other))
        else:
            return AddScalar(-other)(self)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return EWiseDiv()(self, other)
        else:
            return DivScalar(other)(self)

    def __matmul__(self, other):
        return MatMul()(self, other)

    def matmul(self, other):
        return MatMul()(self, other)

    # def sum(self, axes=None):
    #     return Summation(axes)(self)

    # def broadcast_to(self, shape):
    #     return BroadcastTo(shape)(self)

    # def reshape(self, shape):
    #     return Reshape(shape)(self)

    # def __neg__(self):
    #     return Negate()(self)

    # def transpose(self, axes=None):
    #     return Transpose(axes)(self)

    __radd__ = __add__
    __rmul__ = __mul__
    __rsub__ = __sub__
    __rmatmul__ = __matmul__

class TensorOp(Op):
    def __call__(self, *args):
        return Tensor.make_from_op(self, args)


class EWiseAdd(Op):
    def compute(self, a: Value, b: Value) -> mt.Tensor:
        tensor_a = a.cached_data
        tensor_b = b.cached_data
        result_tensor = mt.ewise_add(tensor_a, tensor_b)
        return result_tensor

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: Value):
        tensor_a = a.cached_data
        result_tensor = mt.add_scalar(tensor_a, self.scalar)
        return result_tensor

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: Value, b: Value):
        tensor_a = a.cached_data
        tensor_b = b.cached_data
        result_tensor = mt.ewise_mul(tensor_a, tensor_b)
        return result_tensor

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: Value):
        tensor_a = a.cached_data
        result_tensor = mt.mul_scalar(tensor_a, self.scalar)
        return result_tensor

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """逐点乘方，用标量做指数"""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: Value) -> mt.Tensor:
        tensor_a = a.cached_data
        result_tensor = mt.power_scalar(tensor_a, self.scalar)
        return result_tensor
        

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return out_grad * self.scalar * a ** (self.scalar - 1)
        


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWisePow(TensorOp):
    """逐点乘方"""

    def compute(self, a: Value, b: Value) -> mt.Tensor:
        tensor_a = a.cached_data
        tensor_b = b.cached_data
        result_tensor = mt.ewise_pow(tensor_a, tensor_b)
        return result_tensor

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], Tensor) or not isinstance(
            node.inputs[1], Tensor
        ):
            raise ValueError("Both inputs must be tensors.")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * log(a)
        return grad_a, grad_b

def power(a, b):
    return EWisePow()(a, b)


class EWiseDiv(TensorOp):
    """逐点相除"""

    def compute(self, a, b):
        tensor_a = a.cached_data
        tensor_b = b.cached_data
        result_tensor = mt.ewise_div(tensor_a, tensor_b)
        return result_tensor
        

    def gradient(self, out_grad, node):
        a, b = node.inputs[0], node.inputs[1]
        return out_grad / b, out_grad * ( -a / b ** 2)
        


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        tensor_a = a.cached_data
        result_tensor = mt.div_scalar(tensor_a)
        return result_tensor
        

    def gradient(self, out_grad, node):
        return out_grad / self.scalar
        


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


# class Transpose(TensorOp):
#     def __init__(self, axes: Optional[tuple] = None):
#         self.axes = axes

#     def compute(self, a):
#         if self.axes is None:
#             new_axes = [x for x in range(len(a.shape))]
#             new_axes[-1], new_axes[-2] = new_axes[-2], new_axes[-1]
#             return np.transpose(a, axes = tuple(new_axes))
#         else:
#             d1, d2 = self.axes[0], self.axes[1]
#             new_axes = [x for x in range(len(a.shape))]
#             new_axes[d1], new_axes[d2] = new_axes[d2], new_axes[d1]
#             return np.transpose(a, axes = tuple(new_axes))

        

#     def gradient(self, out_grad, node):
#         return out_grad.transpose(self.axes)
        


# def transpose(a, axes=None):
#     return Transpose(axes)(a)


# class Reshape(TensorOp):
#     def __init__(self, shape):
#         self.shape = shape

#     def compute(self, a):
#         return np.reshape(a, self.shape)
        

#     def gradient(self, out_grad, node):
#         return out_grad.reshape(node.inputs[0].shape)
        


# def reshape(a, shape):
#     return Reshape(shape)(a)


# class BroadcastTo(TensorOp):
#     def __init__(self, shape):
#         self.shape = shape

#     def compute(self, a):
#         return np.broadcast_to(a, self.shape)
        

#     def gradient(self, out_grad, node):
#         shape1 = list(node.inputs[0].shape)
#         shape2 = list(self.shape)
#         axes = []
#         for i in range(1,len(shape2)+1):
#             if i <= len(shape1) and shape2[-i] != shape1[-i]:
#                 axes.append(len(shape2) - i)
#             if i > len(shape1):
#                 axes.append(len(shape2) - i)
#         axes.reverse()
#         return out_grad.sum(tuple(axes)).reshape(shape1)
        


# def broadcast_to(a, shape):
#     return BroadcastTo(shape)(a)


# class Summation(TensorOp):
#     def __init__(self, axes: Optional[tuple] = None):
#         self.axes = axes

#     def compute(self, a):
#         if self.axes is None:
#             return np.sum(a)
#         else:
#             return np.sum(a, axis=self.axes)
        

#     def gradient(self, out_grad: Tensor, node):
#         if self.axes is None:
#             return out_grad.broadcast_to(node.inputs[0].shape)
#         shape1 = list(node.inputs[0].shape)
#         for i in self.axes:
#             shape1[i] = 1
#         return out_grad.reshape(shape1).broadcast_to(node.inputs[0].shape)
        


# def summation(a, axes=None):
#     return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        tensor_a = a.cached_data
        tensor_b = b.cached_data
        result_tensor = mt.matmul(tensor_a, tensor_b)
        return result_tensor
        

    def gradient(self, out_grad, node):
        a, b = node.inputs[0], node.inputs[1]
        la, lb = len(a.shape), len(b.shape)
        if la == lb:
            return out_grad @ b.transpose(), a.transpose() @ out_grad
        elif la > lb:
            axes = [x for x in range(la - 2)]
            axes = tuple(axes)
            return out_grad @ b.transpose(), (a.transpose() @ out_grad).sum(axes)
        else:
            axes = [x for x in range(lb - 2)]
            axes = tuple(axes)
            return (out_grad @ b.transpose()).sum(axes), a.transpose() @ out_grad

        


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        tensor_a = a.cached_data
        result_tensor = mt.negate(tensor_a)
        return result_tensor
        

    def gradient(self, out_grad, node):
        return -out_grad
        


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        tensor_a = a.cached_data
        result_tensor = mt.log(tensor_a)
        return result_tensor
        

    def gradient(self, out_grad, node):
        return out_grad / node.inputs[0]
        


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        tensor_a = a.cached_data
        result_tensor = mt.exp(tensor_a)
        return result_tensor
        

    def gradient(self, out_grad, node):
        return out_grad * exp(node.inputs[0])
        


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        output = mt.Tensor(a.shape,mt.Device.GPU)
        mt.forward_relu(a, output)
        return output
        

    def gradient(self, out_grad, node):
        x = node.inputs[0].numpy()
        x = (x > 0).astype(x.dtype)
        x = Tensor(x, dtype=out_grad.dtype,device= out_grad.device)
        return out_grad * x
        


def relu(a):
    return ReLU()(a)


