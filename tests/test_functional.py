import torch
import unittest
import sys
sys.path.append("..")

from functional import linear, relu, sigmoid, conv2d, max_pool2d, conv_transpose2d


class TestFunctional(unittest.TestCase):
    def test_linear(self):
        x = torch.rand((3, 2))
        weight = torch.rand((4, 2))
        bias = torch.rand(4)
        torch_out = torch.nn.functional.linear(x, weight, bias)
        out = linear(x, weight, bias)
        self.assertTrue(torch.allclose(torch_out, out))
        self.assertEqual(torch_out.shape, out.shape)
    
    def test_relu(self):
        x = torch.rand((3, 2))
        torch_out = torch.nn.functional.relu(x)
        out = relu(x)
        self.assertTrue(torch.allclose(torch_out, out))
        self.assertEqual(torch_out.shape, out.shape)
    
    def test_sigmoid(self):
        x = torch.rand((3, 2))
        torch_out = torch.sigmoid(x)
        out = sigmoid(x)
        self.assertTrue(torch.allclose(torch_out, out))
        self.assertEqual(torch_out.shape, out.shape)

    def test_conv2d_simple(self):
        x = torch.tensor([[[[0., 1., 2., 3.], [4., 5., 6., 7.], [8., 9., 10., 11.],
                            [12., 13., 14., 15.]]]])
        kernels = torch.tensor([[[[0, 1], [2, 3]]]]).float()
        bias = torch.rand(1)
        gt = torch.nn.functional.conv2d(x, kernels, stride=1, dilation=1, padding=0, bias=bias)
        out = conv2d(x, kernels, stride=1, dilation=1, padding=0, bias=bias)
        self.assertTrue(torch.isclose(out, gt).all())

    def test_conv2d_single_input(self):
        x = torch.rand((1, 1, 3, 3))
        kernels = torch.rand((1, 1, 2, 2))
        gt = torch.nn.functional.conv2d(x, kernels, stride=1, dilation=1, padding=0)
        out = conv2d(x, kernels, stride=1, dilation=1, padding=0)
        self.assertTrue(torch.isclose(out, gt).all())

    def test_conv2d_in_channels(self):
        x = torch.rand((1, 3, 5, 8))
        kernels = torch.rand((1, 3, 3, 3))
        gt = torch.nn.functional.conv2d(x, kernels, stride=1, dilation=1, padding=0)
        out = conv2d(x, kernels, stride=1, dilation=1, padding=0)
        self.assertTrue(torch.isclose(out, gt).all())

    def test_conv2d_out_channels(self):
        x = torch.rand((10, 3, 10, 10))
        kernels = torch.rand((4, 3, 2, 2))
        gt = torch.nn.functional.conv2d(x, kernels, stride=1, dilation=1, padding=0)
        out = conv2d(x, kernels, stride=1, dilation=1, padding=0)
        self.assertTrue(torch.isclose(out, gt).all())

    def test_conv2d_dilation(self):
        x = torch.rand((10, 3, 10, 10))
        kernels = torch.rand((4, 3, 2, 2))
        gt = torch.nn.functional.conv2d(x, kernels, stride=1, dilation=5, padding=0)
        out = conv2d(x, kernels, stride=1, dilation=5, padding=0)
        self.assertTrue(torch.isclose(out, gt).all())

    def test_conv2d_stride(self):
        x = torch.rand((10, 3, 10, 10))
        kernels = torch.rand((4, 3, 2, 2))
        gt = torch.nn.functional.conv2d(x, kernels, stride=3, dilation=1, padding=0)
        out = conv2d(x, kernels, stride=3, dilation=1, padding=0)
        self.assertTrue(torch.isclose(out, gt).all())

    def test_conv2d_padding(self):
        x = torch.rand((10, 3, 20, 10))
        kernels = torch.rand((4, 3, 2, 2))
        gt = torch.nn.functional.conv2d(x, kernels, stride=1, dilation=1, padding=4)
        out = conv2d(x, kernels, stride=1, dilation=1, padding=4)
        self.assertTrue(torch.isclose(out, gt).all())

    def test_max_pool2d(self):
        x = torch.rand((2, 3, 4, 4))
        torch_out = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=1)
        out = max_pool2d(x, kernel_size=2, stride=1)
        self.assertTrue(torch.isclose(torch_out, out).all())

        x = torch.rand((2, 3, 32, 32))
        torch_out = torch.nn.functional.max_pool2d(x, kernel_size=4, stride=2, padding=2, dilation=4)
        out = max_pool2d(x, kernel_size=4, stride=2, padding=2, dilation=4)
        self.assertTrue(torch.isclose(torch_out, out).all())

    def test_max_pool2d_padding(self):
        x = torch.rand((2, 3, 16, 16))
        torch_out = torch.nn.functional.max_pool2d(x, kernel_size=4, padding=2)
        out = max_pool2d(x, kernel_size=4, padding=2)
        self.assertTrue(torch.isclose(torch_out, out).all())

        x = torch.rand((2, 3, 16, 16))
        torch_out = torch.nn.functional.max_pool2d(x, kernel_size=4, padding=(1, 2))
        out = max_pool2d(x, kernel_size=4, padding=(1, 2))
        self.assertTrue(torch.isclose(torch_out, out).all())

    def test_max_pool2d_stride(self):
        x = torch.rand((2, 3, 4, 4))
        torch_out = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        out = max_pool2d(x, kernel_size=2, stride=2)
        self.assertTrue(torch.isclose(torch_out, out).all())

        x = torch.rand((2, 3, 4, 4))
        torch_out = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=(2, 4))
        out = max_pool2d(x, kernel_size=2, stride=(2, 4))
        self.assertTrue(torch.isclose(torch_out, out).all())

        x = torch.rand((2, 3, 4, 4))
        torch_out = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=None)
        out = max_pool2d(x, kernel_size=2, stride=None)
        self.assertTrue(torch.isclose(torch_out, out).all())

    def test_max_pool2d_dilation(self):
        x = torch.rand((2, 3, 4, 4))
        torch_out = torch.nn.functional.max_pool2d(x, kernel_size=2, dilation=2)
        out = max_pool2d(x, kernel_size=2, dilation=2)
        self.assertTrue(torch.isclose(torch_out, out).all())

        x = torch.rand((2, 3, 16, 16))
        torch_out = torch.nn.functional.max_pool2d(x, kernel_size=4, dilation=(1, 2))
        out = max_pool2d(x, kernel_size=4, dilation=(1, 2))
        self.assertTrue(torch.isclose(torch_out, out).all())

    def test_conv2d_transpose(self):
        x = torch.rand((4, 3, 8, 8))
        weight = torch.rand((3, 4, 2, 2))
        bias = torch.rand(4)
        gt = torch.nn.functional.conv_transpose2d(x, weight, stride=1, dilation=1, padding=0, bias=bias)
        out = conv_transpose2d(x, weight, stride=1, dilation=1, padding=0, bias=bias)
        self.assertTrue(torch.isclose(out, gt).all())