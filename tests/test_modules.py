import torch
from torch import autograd
import unittest
import sys

sys.path.append("..")

from modules import Linear, Conv2d, TransposeConv2d, ReLU, Sigmoid, Sequential, MSE, MaxPool2d, Upsampling
# from tensor import GTensor
from parameter import Parameter

class TestModules(unittest.TestCase):
    def test_linear(self):
        x = torch.rand((3, 2), requires_grad=True)
        torch_linear = torch.nn.Linear(2, 4)
        linear = Linear(2, 4)
        linear.weight = Parameter(torch_linear.weight.data)
        linear.bias = Parameter(torch_linear.bias.data)
        torch_out = torch_linear(x)
        out = linear(x)
        self.assertTrue(torch.allclose(torch_out, out))
        self.assertEqual(torch_out.shape, out.shape)

    def test_linear_backward(self):
        x = torch.rand((3, 2), requires_grad=True)
        torch_linear = torch.nn.Linear(2, 4)
        linear = Linear(2, 4)
        linear.weight = Parameter(torch_linear.weight.data)
        linear.bias = Parameter(torch_linear.bias.data)
        torch_out = torch_linear(x)
        out = linear(x)
        torch.autograd.backward(torch_out, torch.ones_like(torch_out))
        grad = linear.backward(torch.ones_like(out))
        self.assertEqual(x.grad.shape, grad.shape)
        self.assertEqual(torch_linear.weight.grad.shape, linear.weight.gradient.shape)
        self.assertEqual(torch_linear.bias.grad.shape, linear.bias.gradient.shape)
        self.assertTrue(torch.allclose(x.grad, grad))
        self.assertTrue(torch.allclose(torch_linear.weight.grad, linear.weight.gradient))
        self.assertTrue(torch.allclose(torch_linear.bias.grad, linear.bias.gradient))

    def test_conv2d(self):
        # Create a 1x1x3x3 input tensor
        x = torch.tensor([[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]]])
        # apply convolution
        torch_conv = torch.nn.Conv2d(1, 1, kernel_size=2, stride=1, padding=0, bias=True)
        torch_conv.weight.data = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]])
        torch_conv.bias.data = torch.tensor([5.0])
        # apply kernel
        torch_y = torch_conv(x)
        # apply convolution with our implementation
        conv = Conv2d(1, 1, kernel_size=2, stride=1, padding=0, bias=True)
        conv.weight = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]])
        conv.bias = torch.tensor([5.0])
        # apply kernel
        y = conv(x)
        self.assertTrue(torch.equal(y, torch_y))

    def test_conv2d_backward(self):
        x = torch.tensor([[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]]],
                         requires_grad=True)
        # apply convolution using torch
        torch_conv = torch.nn.Conv2d(1, 1, kernel_size=2, stride=1, padding=0, bias=True)
        torch_conv.weight.data = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]])
        torch_conv.bias.data = torch.tensor([5.0])
        torch_y = torch_conv(x)
        # autograd backwards
        autograd.backward(torch_y, torch.ones_like(torch_y))
        torch_x_grad = x.grad.clone()
        # zero gradients
        x.grad.zero_()
        # apply convolution with our implementation
        conv = Conv2d(1, 1, kernel_size=2, stride=1, padding=0, bias=True)
        conv.weight = Parameter(torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]]))
        conv.bias = Parameter(torch.tensor([5.0]))
        y = conv(x)
        # our backwards
        our_x_grad = conv.backward(torch.ones_like(y))
        # compare gradients
        self.assertTrue(torch.equal(our_x_grad, torch_x_grad))

    def test_conv_transpose2d(self):
        # Create a 1x1x3x3 input tensor
        x = torch.tensor([[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]]])
        # apply transpose convolution
        torch_conv = torch.nn.ConvTranspose2d(1, 1, kernel_size=2, stride=1, padding=1,
                                              bias=True)
        torch_conv.weight.data = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]])
        torch_conv.bias.data = torch.tensor([5.0])
        # apply kernel
        torch_y = torch_conv(x)
        # apply transpose convolution with our implementation
        conv = TransposeConv2d(1, 1, kernel_size=2, stride=1, padding=1, bias=True)
        conv.weight = Parameter(torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]]))
        conv.bias = Parameter(torch.tensor([5.0]))
        # apply kernel
        y = conv(x)
        self.assertTrue(torch.equal(y, torch_y))

    def test_upsampling(self):
        # Create a 1x1x3x3 input tensor
        x = torch.tensor([[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]]])
        # apply transpose convolution
        torch_conv = torch.nn.ConvTranspose2d(1, 1, kernel_size=2, stride=1, padding=0,
                                              bias=True)
        torch_conv.weight.data = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]])
        torch_conv.bias.data = torch.tensor([5.0])
        # apply kernel
        torch_y = torch_conv(x)
        # apply transpose convolution with our implementation
        conv = Upsampling(1, 1, kernel_size=2, stride=1, padding=0, bias=True)
        conv.weight = Parameter(torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]]))
        conv.bias = Parameter(torch.tensor([5.0]))
        # apply kernel
        y = conv(x)
        self.assertTrue(torch.equal(y, torch_y))

    def test_conv_transpose2d_backward(self):
        x = torch.tensor([[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]]],
                         requires_grad=True)
        # apply transpose convolution using torch
        torch_conv = torch.nn.ConvTranspose2d(1, 1, kernel_size=2, stride=1, padding=0,
                                              bias=True)
        torch_conv.weight.data = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]])
        torch_conv.bias.data = torch.tensor([5.0])
        torch_y = torch_conv(x)
        # autograd backwards
        autograd.backward(torch_y, torch.ones_like(torch_y))
        torch_x_grad = x.grad.clone()
        # zero gradients
        x.grad.zero_()
        # apply transpose convolution with our implementation
        conv = TransposeConv2d(1, 1, kernel_size=2, stride=1, padding=0, bias=True)
        conv.weight = Parameter(torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]]))
        conv.bias = Parameter(torch.tensor([5.0]))
        y = conv(x)
        # our backwards
        our_x_grad = conv.backward(torch.ones_like(y))
        # compare gradients
        self.assertTrue(torch.equal(our_x_grad, torch_x_grad))

    def test_relu(self):
        x = torch.rand((3, 2), requires_grad=True)
        torch_relu = torch.nn.ReLU()
        relu = ReLU()
        torch_out = torch_relu(x)
        out = relu(x)
        self.assertTrue(torch.allclose(torch_out, out))
        self.assertEqual(torch_out.shape, out.shape)

    def test_relu_backward(self):
        x = torch.rand((3, 2), requires_grad=True)
        torch_relu = torch.nn.ReLU()
        relu = ReLU()
        torch_out = torch_relu(x)
        out = relu(x)
        torch.autograd.backward(torch_out, torch.ones_like(torch_out))
        grad = relu.backward(torch.ones_like(out))
        self.assertEqual(x.grad.shape, grad.shape)
        self.assertTrue(torch.allclose(x.grad, grad))

    def test_sigmoid(self):
        x = torch.rand((3, 2), requires_grad=True)
        torch_sigmoid = torch.nn.Sigmoid()
        sigmoid = Sigmoid()
        torch_out = torch_sigmoid(x)
        out = sigmoid(x)
        self.assertTrue(torch.allclose(torch_out, out))
        self.assertEqual(torch_out.shape, out.shape)

    def test_sigmoid_backward(self):
        x = torch.rand((3, 2), requires_grad=True)
        torch_sigmoid = torch.nn.Sigmoid()
        sigmoid = Sigmoid()
        torch_out = torch_sigmoid(x)
        out = sigmoid(x)
        torch.autograd.backward(torch_out, torch.ones_like(torch_out))
        grad = sigmoid.backward(torch.ones_like(out))
        self.assertEqual(x.grad.shape, grad.shape)
        self.assertTrue(torch.allclose(x.grad, grad))

    def test_sequential(self):
        x = torch.rand((3, 2), requires_grad=True)
        torch_linear1 = torch.nn.Linear(2, 4)
        torch_linear2 = torch.nn.Linear(4, 1)
        torch_model = torch.nn.Sequential(torch_linear1, torch.nn.Sigmoid(), torch_linear2, torch.nn.ReLU())
        linear1 = Linear(2, 4)
        linear2 = Linear(4, 1)
        linear1.weight = Parameter(torch_linear1.weight.data)
        linear1.bias = Parameter(torch_linear1.bias.data)
        linear2.weight = Parameter(torch_linear2.weight.data)
        linear2.bias = Parameter(torch_linear2.bias.data)
        model = Sequential(linear1, Sigmoid(), linear2, ReLU())
        torch_out = torch_model(x)
        out = model(x)
        self.assertTrue(torch.allclose(torch_out, out))
        self.assertEqual(torch_out.shape, out.shape)

    def test_sequential_backward(self):
        x = torch.rand((3, 2), requires_grad=True)
        torch_linear1 = torch.nn.Linear(2, 4)
        torch_linear2 = torch.nn.Linear(4, 1)
        torch_model = torch.nn.Sequential(torch_linear1, torch.nn.Sigmoid(), torch_linear2, torch.nn.ReLU())
        linear1 = Linear(2, 4)
        linear2 = Linear(4, 1)
        linear1.weight = Parameter(torch_linear1.weight.data)
        linear1.bias = Parameter(torch_linear1.bias.data)
        linear2.weight = Parameter(torch_linear2.weight.data)
        linear2.bias = Parameter(torch_linear2.bias.data)
        model = Sequential(linear1, Sigmoid(), linear2, ReLU())
        torch_out = torch_model(x)
        out = model(x)
        torch.autograd.backward(torch_out, torch.ones_like(torch_out))
        grad = model.backward(torch.ones_like(out))
        self.assertEqual(x.grad.shape, grad.shape)
        self.assertTrue(torch.allclose(x.grad, grad))

    def test_mseloss(self):
        x = torch.rand((3, 4))
        y = torch.rand((3, 4))
        torch_loss = torch.nn.MSELoss()
        loss = MSE()
        torch_out = torch_loss(x, y)
        out = loss(x, y)
        self.assertTrue(torch.allclose(torch_out, out))
        self.assertEqual(torch_out.shape, out.shape)

        torch_loss = torch.nn.MSELoss(reduction="sum")
        loss = MSE(reduction="sum")
        torch_out = torch_loss(x, y)
        out = loss(x, y)
        self.assertTrue(torch.allclose(torch_out, out))
        self.assertEqual(torch_out.shape, out.shape)

        torch_loss = torch.nn.MSELoss(reduction='none')
        loss = MSE(reduction='none')
        torch_out = torch_loss(x, y)
        out = loss(x, y)
        self.assertTrue(torch.allclose(torch_out, out))
        self.assertEqual(torch_out.shape, out.shape)

    def test_mseloss_backward(self):
        x = torch.rand((3, 4), requires_grad=True)
        y = torch.rand((3, 4), requires_grad=True)
        torch_loss = torch.nn.MSELoss()
        loss = MSE()
        torch_out = torch_loss(x, y)
        out = loss(x, y)
        torch.autograd.backward(torch_out, torch.ones_like(torch_out))
        grad = loss.backward(torch.ones_like(out))
        self.assertEqual(x.grad.shape, grad.shape)
        self.assertTrue(torch.allclose(x.grad, grad))

        x = torch.rand((3, 4), requires_grad=True)
        y = torch.rand((3, 4), requires_grad=True)
        torch_loss = torch.nn.MSELoss(reduction="sum")
        loss = MSE(reduction="sum")
        torch_out = torch_loss(x, y)
        out = loss(x, y)
        torch.autograd.backward(torch_out, torch.ones_like(torch_out))
        grad = loss.backward(torch.ones_like(out))
        self.assertEqual(x.grad.shape, grad.shape)
        self.assertTrue(torch.allclose(x.grad, grad))

    def test_max_pool2d(self):
        x = torch.rand((2, 3, 4, 4))
        torch_maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=1)
        maxpool = MaxPool2d(kernel_size=2, stride=1)
        torch_out = torch_maxpool(x)
        out = maxpool(x)
        self.assertTrue(torch.isclose(torch_out, out).all())

        x = torch.rand((2, 3, 32, 32))
        torch_maxpool = torch.nn.MaxPool2d(kernel_size=4, stride=2, padding=2, dilation=4)
        maxpool = MaxPool2d(kernel_size=4, stride=2, padding=2, dilation=4)
        torch_out = torch_maxpool(x)
        out = maxpool(x)
        self.assertTrue(torch.isclose(torch_out, out).all())

    def test_maxpool2d_backward(self):
        x = torch.rand((2, 3, 32, 32), requires_grad=True)
        torch_maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=2)
        maxpool = MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=2)
        torch_out = torch_maxpool(x)
        out = maxpool(x)
        self.assertTrue(torch.isclose(torch_out, out).all())

        autograd.backward(torch_out, torch.ones_like(torch_out))
        torch_grad = x.grad
        grad = maxpool.backward(torch.ones_like(out))
        self.assertTrue(torch.isclose(torch_grad, grad).all())
