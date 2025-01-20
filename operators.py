import numpy as np
from typing import List, Optional, Tuple, Union
from device import cpu, Device
from basic_operator import Op, Value
from autodiff import compute_gradient_of_variables
from functools import reduce
import MyTensor as mt

def constant(*shape, c=1.0, device=None, dtype="float32", requires_grad=False):
    """Generate constant Tensor"""
    device = cpu() if device is None else device
    array = mt.tensor_from_numpy(device.ones(*shape, dtype=dtype) * c)  # note: can change dtype
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
        if isinstance(numpy_array, mt.Tensor):
            return numpy_array
        return mt.tensor_from_numpy(numpy_array)
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
        return self.realize_cached_data().shape()

    @property
    def dtype(self):
        return self.realize_cached_data().dtype()

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
        data = self.realize_cached_data().to_numpy()

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


class EWiseAdd(TensorOp):
    def compute(self, a: mt.Tensor, b: mt.Tensor):
        result_tensor = mt.ewise_add(a, b)
        return result_tensor

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: mt.Tensor):
        result_tensor = mt.add_scalar(a, self.scalar)
        return result_tensor

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: mt.Tensor, b: mt.Tensor):
        result_tensor = mt.ewise_mul(a, b)
        return result_tensor

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: mt.Tensor):
        result_tensor = mt.mul_scalar(a, self.scalar)
        return result_tensor

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """逐点乘方，用标量做指数"""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: mt.Tensor) -> mt.Tensor:
        result_tensor = mt.power_scalar(a, self.scalar)
        return result_tensor
        

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return out_grad * self.scalar * a ** (self.scalar - 1)
        


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWisePow(TensorOp):
    """逐点乘方"""

    def compute(self, a: mt.Tensor, b: mt.Tensor) -> mt.Tensor:
        result_tensor = mt.ewise_pow(a, b)
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
        result_tensor = mt.ewise_div(a, b)
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
        result_tensor = mt.div_scalar(a, self.scalar)
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


# class MatMul(TensorOp):
#     def compute(self, a, b):
#         result_tensor = mt.matmul(a, b)
#         return result_tensor
        

#     def gradient(self, out_grad, node):
#         a, b = node.inputs[0], node.inputs[1]
#         la, lb = len(a.shape), len(b.shape)
#         if la == lb:
#             return out_grad @ b.transpose(), a.transpose() @ out_grad
#         elif la > lb:
#             axes = [x for x in range(la - 2)]
#             axes = tuple(axes)
#             return out_grad @ b.transpose(), (a.transpose() @ out_grad).sum(axes)
#         else:
#             axes = [x for x in range(lb - 2)]
#             axes = tuple(axes)
#             return (out_grad @ b.transpose()).sum(axes), a.transpose() @ out_grad

        


# def matmul(a, b):
#     return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        result_tensor = mt.negate(a)
        return result_tensor
        

    def gradient(self, out_grad, node):
        return -out_grad
        


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        result_tensor = mt.log(a)
        return result_tensor
        

    def gradient(self, out_grad, node):
        return out_grad / node.inputs[0]
        


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        result_tensor = mt.exp(a)
        return result_tensor
        

    def gradient(self, out_grad, node):
        return out_grad * exp(node.inputs[0])
        


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        output = mt.Tensor(a.shape(),mt.Device.GPU)
        mt.forward_relu(a, output)
        return output
        

    def gradient(self, out_grad, node):
        x = node.inputs[0].numpy()
        x = (x > 0).astype(x.dtype)
        x = Tensor(x)
        return out_grad * x
        


def relu(a):
    return ReLU()(a)



class Linear(TensorOp):
    def compute(self, input, weight):
        self.batch_size = input.shape()[0]
        self.in_features = input.shape()[1]
        self.out_features = weight.shape()[0]
        output = mt.Tensor([self.batch_size, self.out_features], mt.Device.GPU)
        mt.forward_fc(input, output, weight, self.batch_size, self.in_features, self.out_features)
        return output
    
    def gradient(self, out_grad, node):
        input = node.inputs[0].realize_cached_data()
        weight = node.inputs[1].realize_cached_data()
        in_grad = mt.Tensor([self.batch_size, self.in_features],mt.Device.GPU)
        weight_grad = mt.Tensor([self.out_features, self.in_features], mt.Device.GPU)
        mt.backward_fc(input, weight, self.batch_size, self.in_features, self.out_features, out_grad.realize_cached_data(), in_grad, weight_grad)
        # print(np.max(input.to_numpy()), np.min(input.to_numpy()),np.shape(input.to_numpy()))
        # print(np.max(weight.to_numpy()), np.min(weight.to_numpy()),np.shape(weight.to_numpy())) 
        # print(self.batch_size, self.in_features, self.out_features) 
        # print(np.max(out_grad.numpy()), np.min(out_grad.numpy()),np.shape(out_grad.numpy()))
        # print(np.max(in_grad.to_numpy()), np.min(in_grad.to_numpy()),np.shape(in_grad.to_numpy()))
        # print(np.max(weight_grad.to_numpy()), np.min(weight_grad.to_numpy()),np.shape(weight_grad.to_numpy())) 
        in_grad = Tensor(in_grad)
        weight_grad = Tensor(weight_grad)
        return in_grad, weight_grad


def linear(input, weight):
    return Linear()(input, weight)


class Conv(TensorOp):
    def compute(self, input, weight):
        self.batch_size = 1
        self.in_features = 1
        self.out_features = weight.shape()[0]
        self.height = input.shape()[0]
        self.width = input.shape()[1]
        self.column = mt.Tensor([self.batch_size, self.width*self.height, 9*self.in_features], mt.Device.GPU)
        output = mt.Tensor([self.batch_size, self.out_features, self.height, self.width], mt.Device.GPU)
        mt.forward_conv(input, self.column, weight, output, self.batch_size, self.in_features, self.out_features, self.height, self.width)
        return output
    
    def gradient(self, out_grad, node):
        weight = node.inputs[1].realize_cached_data()
        in_grad = mt.Tensor([self.batch_size, self.in_features, self.height, self.width], mt.Device.GPU)
        weight_grad = mt.Tensor([self.out_features, self.in_features, 3, 3], mt.Device.GPU)
        column_grad = mt.Tensor([self.batch_size, self.width*self.height, 9*self.in_features], mt.Device.GPU)
        mt.backward_conv(self.column, column_grad, weight, self.batch_size, self.in_features, self.out_features, self.height, self.width, out_grad.realize_cached_data(), in_grad, weight_grad)
        in_grad = Tensor(in_grad)
        weight_grad = Tensor(weight_grad)
        return in_grad, weight_grad
    
def conv(input, weight):
    return Conv()(input, weight)


class MaxPool(TensorOp):
    def compute(self, input):
        self.batch_size = input.shape()[0]
        self.in_features = input.shape()[1]
        self.height = input.shape()[2]
        self.width = input.shape()[3]
        output = mt.Tensor([self.batch_size, self.in_features, self.height//2, self.width//2], mt.Device.GPU)
        self.mask = mt.Tensor([self.batch_size, self.in_features, self.height, self.width], mt.Device.GPU)
        mt.forward_maxpool(input, output, self.mask, self.batch_size, self.in_features, self.height, self.width)
        return output
    
    def gradient(self, out_grad, node):
        in_grad = mt.Tensor([self.batch_size, self.in_features, self.height*2, self.width*2], mt.Device.GPU)
        mt.backward_maxpool(out_grad.cached_data, in_grad, self.mask, self.batch_size, self.in_features, self.height, self.width)
        in_grad = Tensor(in_grad)
        return in_grad
    
def maxpool(input):
    return MaxPool()(input)


class CrossEntropy(TensorOp):
    def compute(self, x, label):
        self.batch_size = x.shape()[0]
        self.in_features = x.shape()[1]
        self.output = mt.Tensor([self.batch_size, self.in_features], mt.Device.GPU)
        loss = mt.Tensor([self.batch_size], mt.Device.GPU)
        mt.forward_softmax(x, self.output, self.batch_size, self.in_features)
        mt.forward_cross_entropy(self.output, loss, label, self.batch_size, self.in_features)
        return loss
    
    def gradient(self, out_grad, node):
        label = node.inputs[1].realize_cached_data()
        grad_loss = mt.Tensor([self.batch_size, self.in_features], mt.Device.GPU)
        mt.backward_cross_entropy(self.output, grad_loss, label, self.batch_size, self.in_features)
        grad_loss = Tensor(grad_loss)
        label = Tensor(label)
        return grad_loss, label
    
def cross_entropy(output, label):
    return CrossEntropy()(output, label)


class Flatten(TensorOp):
    def compute(self, input):
        self.shape = input.shape()
        output = mt.reshape(input, [1, reduce(lambda x, y: x * y, self.shape)])
        return output
    
    def gradient(self, out_grad, node):
        in_grad = mt.reshape(out_grad.realize_cached_data(), self.shape)
        in_grad = Tensor(in_grad)
        return in_grad
    
def flatten(input):
    return Flatten()(input)
