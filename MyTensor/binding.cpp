#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "Tensor.h"
#include "Module.h"

namespace py = pybind11;

Tensor tensor_from_numpy(py::array_t<float> data){
    std::vector<int> shape(data.ndim());
    for (int i = 0; i < shape.size(); i++){
        shape[i] = data.shape(i);
    }
    Tensor tensor(shape, Device::CPU);
    for (int i = 0; i < tensor.length(); i++){
        tensor.data()[i] = data.data()[i];
    }
    return tensor;
}

PYBIND11_MODULE(MyTensor, m) {
    // 绑定枚举类型
    py::enum_<Device>(m, "Device")
        .value("CPU", Device::CPU)
        .value("GPU", Device::GPU)
        .export_values();

    // 绑定 Tensor 类
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<const std::vector<int> &, Device>())
        .def("cpu", &Tensor::cpu)
        .def("gpu", &Tensor::gpu)
        .def("assign", &Tensor::assign)
        .def("Print", &Tensor::Print)
        .def("to_numpy", [](Tensor &tensor) {
                // 转numpy时内存转cpu，统一内存管理
                tensor = tensor.cpu();
                return py::array_t<float>(tensor.shape_, tensor.data());
             })
        .def("shape",[](Tensor &tensor) {return tensor.shape_;})
        .def("dtype",[](Tensor &tensor) {return tensor.device_;});

    // 绑定 Module 函数
    m.def("forward_sigmoid", &Sigmoid)
        .def("backward_sigmoid", &Sigmoid_backward)
        .def("forward_relu", &Relu)
        .def("backward_relu", &Relu_backward)
        .def("forward_fc", &forward_fc)
        .def("backward_fc", &backward_fc)
        .def("forward_conv", &forward_conv)
        .def("backward_conv", &backward_conv)
        .def("forward_maxpool", &forward_maxpool)
        .def("backward_maxpool", &backward_maxpool)
        .def("forward_softmax", &forward_softmax)
        .def("forward_cross_entropy", &forward_cross_entropy)
        .def("backward_cross_entropy", &backward_cross_entropy)
        .def("tensor_from_numpy", &tensor_from_numpy)
        .def("ewise_add", &EWiseAdd)
        .def("add_scalar", &AddScalar)
        .def("ewise_mul", &EWiseMul)
        .def("mul_scalar", &MulScalar)
        .def("power_scalar", &PowerScalar)
        .def("ewise_pow", &EWisePow)
        .def("ewise_div", &EWiseDiv)
        .def("div_scalar", &DivScalar)
        .def("negate", &Negate)
        .def("log", &Log)
        .def("exp", &Exp)
        .def("matmul", &Matmul)
        .def("reshape", &Reshape);

}
