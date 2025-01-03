import unittest
import numpy as np
import torch
import torch.nn.functional as F
from MyTensor import Tensor, Device, tensor_from_numpy, forward_fc, backward_fc, forward_conv, backward_conv, forward_maxpool, backward_maxpool, forward_softmax, forward_cross_entropy, backward_cross_entropy, forward_sigmoid, backward_sigmoid, forward_relu, backward_relu

class TestOperators(unittest.TestCase):
    def test_forward_fc(self):
        batch_size, in_features, out_features = 4, 8, 16
        input_data = np.random.rand(batch_size, in_features).astype(np.float32)
        weight_data = np.random.rand(out_features, in_features).astype(np.float32)
        
        input_tensor = tensor_from_numpy(input_data).gpu()
        weight_tensor = tensor_from_numpy(weight_data).gpu()
        output_tensor = Tensor([batch_size, out_features], Device.GPU)
        
        forward_fc(input_tensor, output_tensor, weight_tensor, batch_size, in_features, out_features)
        
        input_torch = torch.tensor(input_data)
        weight_torch = torch.tensor(weight_data)
        output_torch = input_torch @ weight_torch.T
        
        np.testing.assert_allclose(output_tensor.cpu().to_numpy(), output_torch.numpy(), atol=1e-5)

    def test_backward_fc(self):
        batch_size, in_features, out_features = 4, 8, 16
        input_data = np.random.rand(batch_size, in_features).astype(np.float32)
        weight_data = np.random.rand(out_features, in_features).astype(np.float32)
        grad_output_data = np.random.rand(batch_size, out_features).astype(np.float32)
        
        input_tensor = tensor_from_numpy(input_data).gpu()
        weight_tensor = tensor_from_numpy(weight_data).gpu()
        grad_output_tensor = tensor_from_numpy(grad_output_data).gpu()
        
        grad_input_tensor = Tensor([batch_size, in_features], Device.GPU)
        grad_weight_tensor = Tensor([out_features, in_features], Device.GPU)
        
        input_torch = torch.tensor(input_data, requires_grad=True)
        weight_torch = torch.tensor(weight_data, requires_grad=True)
        grad_output_torch = torch.tensor(grad_output_data)
        
        output_torch = input_torch @ weight_torch.T
        output_torch.backward(grad_output_torch)
        output_tensor = tensor_from_numpy(output_torch.detach().numpy()).gpu()

        backward_fc(input_tensor, output_tensor, weight_tensor, 
                    batch_size, in_features, out_features, grad_output_tensor, grad_input_tensor, grad_weight_tensor)
        
        np.testing.assert_allclose(grad_input_tensor.cpu().to_numpy(), input_torch.grad.numpy(), atol=1e-5)
        np.testing.assert_allclose(grad_weight_tensor.cpu().to_numpy(), weight_torch.grad.numpy(), atol=1e-5)

    def test_forward_conv(self):
        batch_size, in_features, out_features = 2, 3, 6
        height, width = 8, 8
        input_data = np.random.rand(batch_size, in_features, height, width).astype(np.float32)
        weight_data = np.random.rand(out_features, in_features, 3, 3).astype(np.float32)
        
        input_tensor = tensor_from_numpy(input_data).gpu()
        weight_tensor = tensor_from_numpy(weight_data).gpu()
        column_tensor = Tensor([batch_size, width * height, 9 * in_features], Device.GPU)
        output_tensor = Tensor([batch_size, out_features, height, width], Device.GPU)
        
        forward_conv(input_tensor, column_tensor, weight_tensor, output_tensor, 
                     batch_size, in_features, out_features, height, width)
        
        input_torch = torch.tensor(input_data)
        weight_torch = torch.tensor(weight_data)
        output_torch = F.conv2d(input_torch, weight_torch, padding=1)
        
        # 验证结果
        np.testing.assert_allclose(output_tensor.cpu().to_numpy(), output_torch.numpy(), atol=1e-5)

    def test_backward_conv(self):
        batch_size, in_features, out_features = 3, 4, 3
        height, width = 8, 8
        input_data = np.random.rand(batch_size, in_features, height, width).astype(np.float32)
        weight_data = np.random.rand(out_features, in_features, 3, 3).astype(np.float32)
        grad_output_data = np.random.rand(batch_size, out_features, height, width).astype(np.float32)

        input_tensor = tensor_from_numpy(input_data).gpu()
        output_tensor = Tensor([batch_size, out_features, height, width], Device.GPU)
        column_tensor = Tensor([batch_size, height * width, in_features * 9], Device.GPU)
        grad_column_tensor = Tensor([batch_size, height * width, in_features * 9], Device.GPU)
        weight_tensor = tensor_from_numpy(weight_data).gpu()
        grad_output_tensor = tensor_from_numpy(grad_output_data).gpu()
        
        grad_input_tensor = Tensor([batch_size, in_features, height, width], Device.GPU)
        grad_weight_tensor = Tensor([out_features, in_features, 3, 3], Device.GPU)

        forward_conv(input_tensor, column_tensor, weight_tensor, output_tensor, batch_size, in_features, out_features, height, width)
        backward_conv(column_tensor, grad_column_tensor, weight_tensor, batch_size, in_features, out_features, height, width, grad_output_tensor, grad_input_tensor, grad_weight_tensor)
        
        input_torch = torch.tensor(input_data, requires_grad=True)
        weight_torch = torch.tensor(weight_data, requires_grad=True)
        grad_output_torch = torch.tensor(grad_output_data)
        
        output_torch = F.conv2d(input_torch, weight_torch, padding=1)
        output_torch.backward(grad_output_torch)
        
        # 验证结果
        np.testing.assert_allclose(grad_input_tensor.cpu().to_numpy(), input_torch.grad.numpy(), atol=1e-5)
        np.testing.assert_allclose(grad_weight_tensor.cpu().to_numpy(), weight_torch.grad.numpy(), atol=1e-5)
        

    def test_maxpool(self):
        batch_size, in_features, height, width = 2, 3, 6, 7
        input_data = np.random.rand(batch_size, in_features, height, width).astype(np.float32)
        input_tensor = tensor_from_numpy(input_data).gpu()
        output_tensor = Tensor([batch_size, in_features, height // 2, width // 2], Device.GPU)
        mask_tensor = Tensor([batch_size, in_features, height, width], Device.GPU)

        grad_output_data = np.random.rand(batch_size, in_features, height // 2, width // 2).astype(np.float32)
        grad_output_tensor = tensor_from_numpy(grad_output_data).gpu()
        grad_input_tensor = Tensor([batch_size, in_features, height, width], Device.GPU)
        grad_output_torch = torch.tensor(grad_output_data, requires_grad=False)

        forward_maxpool(input_tensor, output_tensor, mask_tensor, batch_size, in_features, height, width)
        backward_maxpool(grad_output_tensor, grad_input_tensor, mask_tensor, batch_size, in_features, height, width)

        # PyTorch 计算参考结果
        input_torch = torch.tensor(input_data, requires_grad=False)
        output_torch, indices_torch = F.max_pool2d(
            input_torch, kernel_size=2, stride=2, return_indices=True
        )
        grad_input_torch = F.max_unpool2d(grad_output_torch, indices_torch, kernel_size=2, stride=2, output_size=input_torch.shape)

        # 验证结果
        np.testing.assert_allclose(
            output_tensor.cpu().to_numpy(), output_torch.numpy(), rtol=1e-5, atol=1e-5,
            err_msg="forward_maxpool failed!" # 前向传播
        )
        np.testing.assert_allclose(
            grad_input_tensor.cpu().to_numpy(), grad_input_torch.numpy(), rtol=1e-5, atol=1e-5,
            err_msg="backward_maxpool failed!" #反向传播
        )

    def test_softmax_entropy(self):
        batch_size, in_features = 4, 10
        input_data = np.random.rand(batch_size, in_features).astype(np.float32)
        input_tensor = tensor_from_numpy(input_data).gpu()

        labels_data = np.random.randint(0, in_features, size=(batch_size)).astype(np.int32)  # 每个样本的真实类别
        labels_tensor = tensor_from_numpy(labels_data).gpu()

        output_tensor = Tensor([batch_size, in_features], Device.GPU)
        forward_softmax(input_tensor, output_tensor, batch_size, in_features)
        input_torch = torch.tensor(input_data)
        output_data = F.softmax(input_torch,1)

        # 验证 forward_softmax 结果
        np.testing.assert_allclose(
            output_tensor.cpu().to_numpy(), output_data, rtol=1e-5, atol=1e-5,
            err_msg="forward_softmax failed!"
        )

        loss_tensor = Tensor([batch_size], Device.GPU)  
        grad_loss_tensor = Tensor([batch_size, in_features], Device.GPU)

        forward_cross_entropy(output_tensor, loss_tensor, labels_tensor, batch_size, in_features)

        true_class_probs = output_data.numpy()[np.arange(batch_size), labels_data]
        expected_loss = -np.mean(np.log(true_class_probs))

        # 验证前向传播损失值
        np.testing.assert_allclose(
            np.mean(loss_tensor.cpu().to_numpy()), expected_loss, rtol=1e-5, atol=1e-5,
            err_msg="forward_cross_entropy failed!"
        )

        grad_logits = output_data.numpy().copy()
        grad_logits[np.arange(batch_size), labels_data] -= 1

        backward_cross_entropy(output_tensor, grad_loss_tensor, labels_tensor, batch_size, in_features)

        # 验证反向传播梯度
        np.testing.assert_allclose(
            grad_loss_tensor.cpu().to_numpy(), grad_logits, rtol=1e-5, atol=1e-5,
            err_msg="backward_cross_entropy failed!"
        )

    def test_sigmoid(self):
        input_data = np.array([[0.5, -0.2], [-0.1, 1.5]], dtype=np.float32)
        grad_output_data = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)

        input_tensor = tensor_from_numpy(input_data).gpu()
        grad_output_tensor = tensor_from_numpy(grad_output_data).gpu()

        output_tensor = Tensor([2,2], Device.GPU)
        forward_sigmoid(input_tensor, output_tensor)

        grad_input_tensor = Tensor([2,2], Device.GPU)
        backward_sigmoid(grad_output_tensor, grad_input_tensor, output_tensor)

        input_torch = torch.tensor(input_data, requires_grad=True)
        output_torch = torch.sigmoid(input_torch)
        output_torch.backward(torch.tensor(grad_output_data))

        # 验证前向传播结果
        np.testing.assert_almost_equal(
            output_tensor.cpu().to_numpy(), output_torch.detach().numpy(), decimal=5
        )

        # 验证反向传播结果
        np.testing.assert_almost_equal(
            grad_input_tensor.cpu().to_numpy(), input_torch.grad.numpy(), decimal=5
        )

    def test_relu(self):
        input_data = np.array([[0.5, -0.2], [-0.1, 1.5]], dtype=np.float32)
        grad_output_data = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)

        input_tensor = tensor_from_numpy(input_data).gpu()
        grad_output_tensor = tensor_from_numpy(grad_output_data).gpu()

        output_tensor = Tensor([2,2], Device.GPU)
        forward_relu(input_tensor, output_tensor)

        grad_input_tensor = Tensor([2,2], Device.GPU)
        backward_relu(grad_output_tensor, grad_input_tensor, input_tensor)

        input_torch = torch.tensor(input_data, requires_grad=True)
        output_torch = F.relu(input_torch)
        output_torch.backward(torch.tensor(grad_output_data))

        # 验证前向传播结果
        np.testing.assert_almost_equal(
            output_tensor.cpu().to_numpy(), output_torch.detach().numpy(), decimal=5
        )

        # 验证反向传播结果
        np.testing.assert_almost_equal(
            grad_input_tensor.cpu().to_numpy(), input_torch.grad.numpy(), decimal=5
        )


if __name__ == "__main__":
    unittest.main()


