from typing import Optional, Tuple, Union
from functools import reduce
import random
import math

import torch
from torch.nn.functional import fold, unfold

try:
    from .autograd import autograd_tensor, accumulate_grad
    from .module import Module
    from .parameter import Parameter
    from .functional import linear, relu, sigmoid, conv2d, conv_transpose2d, max_pool2d
    from .utils import check_inputs, get_gradient, zeros, ones, zeros_like, ones_like
except:
    from autograd import autograd_tensor, accumulate_grad
    from module import Module
    from parameter import Parameter
    from functional import linear, relu, sigmoid, conv2d, conv_transpose2d, max_pool2d
    from utils import check_inputs, get_gradient, zeros, ones, zeros_like, ones_like


class Sequential(Module):
    def __init__(self, *modules) -> None:
        super().__init__('Sequential')
        for module in modules:
            self.register_module(module)

    def forward(self, *input):
        check_inputs(input)
        output = input[0]

        for module in self.modules():
            output = module.forward(output)

        if not self._training:
            return output

        return autograd_tensor(output, self, input[0])

    def backward(self, *gradwrtoutput):
        output = get_gradient(gradwrtoutput)

        for module in self.modules()[::-1]:
            output = module.backward(output)

        return output


class Linear(Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True) -> None:
        """Initialize linear module

        Args:
            in_dim (int): Input dimension
            out_dim (int): Output dimension
            bias (bool, optional): Whether to add bias. Defaults to True.
        """
        super().__init__('Linear')
        self.in_dim = in_dim
        self.out_dim = out_dim
        # TODO: Bad initialization. Implement Xavier initialization
        self.weight = Parameter(torch.empty((out_dim, in_dim)))
        self.bias = Parameter(torch.empty(out_dim)) if bias else None
        self.input_ = None
        self.register_parameter('weight', self.weight)
        self.register_parameter('bias', self.bias)

    def forward(self, *input):
        check_inputs(input)
        input_ = input[0]
        output = linear(input_, self.weight, self.bias)

        if not self._training:
            return output

        self.input_ = input_
        output = autograd_tensor(output, self, self.input_)
        return output

    def backward(self, *gradwrtoutput):
        output_grad = get_gradient(gradwrtoutput)
        weight_grad = output_grad.T.mm(self.input_)
        bias_grad = output_grad.T.mm(ones((self.input_.shape[0], 1))).squeeze()
        input_grad = output_grad.mm(self.weight)
        accumulate_grad(self.weight, weight_grad)

        if self.bias is not None:
            accumulate_grad(self.bias, bias_grad)

        self.input_ = None

        return input_grad


class Conv2d(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple],
                 stride: Union[int, Tuple] = 1, padding: Union[int, Tuple] = 0,
                 groups: int = 1, bias: bool = True, dilation: Union[int, Tuple] = 1) -> None:
        """Initialize a Convolution layer

        Args:
            in_channels (int): Input channels
            out_channels (int): Output channels
            kernel_size (Union[int, Tuple]): Kernel size
            stride (Union[int, Tuple], optional): Stride. Defaults to 1.
            padding (Union[int, Tuple], optional): Padding. Defaults to 0.
            groups (int, optional): Groups. Defaults to 1.
            bias (bool, optional): Bias. Defaults to True.
            dilation (Union[int, Tuple], optional): Dilation. Defaults to 1.
        """
        super().__init__("Conv2d")
        assert len(kernel_size) == 2 if isinstance(kernel_size, tuple) else \
            isinstance(kernel_size, int)

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        assert len(stride) == 2 if isinstance(stride, tuple) else \
            isinstance(stride, int)

        if isinstance(stride, int):
            stride = (stride, stride)

        assert len(padding) == 2 if isinstance(padding, tuple) else \
            isinstance(padding, int)

        if isinstance(padding, int):
            padding = (padding, padding)

        assert len(dilation) == 2 if isinstance(dilation, tuple) else \
            isinstance(dilation, int)

        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.weight = Parameter(
            self.init_weights((self.out_channels, self.in_channels // self.groups,
                               self.kernel_size[0], self.kernel_size[1])))
        self.bias = Parameter(self.init_weights((self.out_channels,))) if bias else None
        self.register_parameter("weight", self.weight)
        self.register_parameter("bias", self.bias)

    def init_weights(self, shape):
        # based on https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        k = self.groups / (self.in_channels * self.kernel_size[0] * self.kernel_size[1])
        w = torch.tensor([random.uniform(-math.sqrt(k), math.sqrt(k)) for _ in range(reduce(lambda a,b: a*b, list(shape)))])
        return w.reshape(shape)

    def forward(self, *input_):
        check_inputs(input_)
        input_ = input_[0]
        output = conv2d(input_, self.weight, self.bias,
                        padding=self.padding, stride=self.stride, dilation=self.dilation)
        
        if not self._training:
            return output

        self.input_ = input_
        output = autograd_tensor(output, self, self.input_)
        return output

    def backward(self, *gradwrtoutput):
        output_grad = get_gradient(gradwrtoutput)

        # Compute bias gradient
        if self.bias is not None:
            bias_grad = output_grad.sum(dim=(0, 2, 3))
            accumulate_grad(self.bias, bias_grad)

        N, C_out, H_out, W_out = output_grad.shape
        N, C_in, H_in, W_in = self.input_.shape

        output_grad = output_grad.transpose(0, 1).reshape(C_out, -1)

        # Compute input gradient
        input_grad = output_grad.T.matmul(self.weight.reshape(C_out, -1))
        input_grad = input_grad.reshape(N, H_out * W_out, C_in * self.kernel_size[0] * self.kernel_size[1]).transpose(1, 2)

        input_grad = fold(input_grad, output_size=(H_in, W_in), kernel_size=self.kernel_size,
                          dilation=self.dilation, padding=self.padding, stride=self.stride)

        # Compute weight gradient
        input_ = unfold(self.input_, kernel_size=self.kernel_size, dilation=self.dilation, padding=self.padding,
                        stride=self.stride)
        input_ = input_.transpose(1, 2).reshape(N * H_out * W_out, -1)
        weight_grad = output_grad.matmul(input_).reshape(self.weight.shape)
        accumulate_grad(self.weight, weight_grad)

        self.input_ = None

        return input_grad


class TransposeConv2d(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple],
                 stride: Union[int, Tuple] = 1, padding: Union[int, Tuple] = 0,
                 groups: int = 1, bias: bool = True, dilation: Union[int, Tuple] = 1) -> None:
        """Initialize Transpose Convolution layer

        Args:
            in_channels (int): Input channels
            out_channels (int): Output channels
            kernel_size (Union[int, Tuple]): Kernel size
            stride (Union[int, Tuple], optional): Stride. Defaults to 1.
            padding (Union[int, Tuple], optional): Padding. Defaults to 0.
            groups (int, optional): Groups. Defaults to 1.
            bias (bool, optional): Bias. Defaults to True.
            dilation (Union[int, Tuple], optional): Dilation. Defaults to 1.
        """
        super().__init__('TransposeConv2d')
        assert len(kernel_size) == 2 if isinstance(kernel_size, tuple) else \
            isinstance(kernel_size, int)

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        assert len(stride) == 2 if isinstance(stride, tuple) else \
            isinstance(stride, int)

        if isinstance(stride, int):
            stride = (stride, stride)

        assert len(padding) == 2 if isinstance(padding, tuple) else \
            isinstance(padding, int)

        if isinstance(padding, int):
            padding = (padding, padding)

        assert len(dilation) == 2 if isinstance(dilation, tuple) else \
            isinstance(dilation, int)

        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.weight = Parameter(self.init_weights((self.in_channels,
                                             self.out_channels // self.groups,
                                             self.kernel_size[0], self.kernel_size[1])))
        self.bias = Parameter(self.init_weights((self.out_channels,))) if bias else None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.input_ = None
        self.register_parameter("weight", self.weight)
        self.register_parameter("bias", self.bias)

    def init_weights(self, shape):
        # based on https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
        k = self.groups / (self.out_channels * self.kernel_size[0] * self.kernel_size[1])
        w = torch.tensor([random.uniform(-math.sqrt(k), math.sqrt(k)) for _ in range(reduce(lambda a,b: a*b, list(shape)))])
        return w.reshape(shape)

    def forward(self, *input):
        check_inputs(input)
        check_inputs(input[0].shape, length=4)
        input_ = input[0]
        output = conv_transpose2d(input_, weight=self.weight, bias=self.bias if self.bias is not None else None,
                                  stride=self.stride, padding=self.padding, dilation=self.dilation)
        
        if not self._training:
            return output

        self.input_ = input_
        output = autograd_tensor(output, self, self.input_)
        return output

    def backward(self, *gradwrtoutput):
        output_grad = get_gradient(gradwrtoutput)

        # Compute bias gradient
        if self.bias is not None:
            bias_grad = output_grad.sum(dim=(0, 2, 3))
            accumulate_grad(self.bias, bias_grad)

        N, C_in, H_in, W_in = self.input_.shape

        output_grad = unfold(output_grad, kernel_size=self.kernel_size, dilation=self.dilation,
                             padding=self.padding, stride=self.stride)
        output_grad = output_grad.transpose(1, 2).reshape(N * H_in * W_in, -1)

        # Compute input gradient
        input_grad = output_grad.matmul(self.weight.reshape(C_in, -1).T)
        input_grad = input_grad.reshape(N, H_in * W_in, C_in).transpose(1, 2).reshape(N, C_in, H_in, W_in)

        # Compute weight gradient
        input_ = self.input_.reshape(C_in, -1)
        weight_grad = input_.matmul(output_grad).reshape(self.weight.shape)
        accumulate_grad(self.weight, weight_grad)

        self.input_ = None

        return input_grad


class ReLU(Module):
    def __init__(self) -> None:
        super().__init__("ReLU")
        self.input_ = None

    def forward(self, *input):
        check_inputs(input)
        input_ = input[0]
        output = relu(input_)

        if not self._training:
            return output

        self.input_ = input_
        output = autograd_tensor(output, self, self.input_)
        return output

    def backward(self, *gradwrtoutput):
        grad = get_gradient(gradwrtoutput)
        input_grad = grad * (self.input_ > 0).int()
        self.input_ = None
        return input_grad


class Sigmoid(Module):
    def __init__(self) -> None:
        super().__init__("Sigmoid")
        self.input_ = None

    def forward(self, *input):
        check_inputs(input)
        input_ = input[0]
        output = sigmoid(input_)

        if not self._training:
            return output

        self.input_ = input_
        output = autograd_tensor(output, self, self.input_)
        return output

    def backward(self, *gradwrtoutput):
        output_grad = get_gradient(gradwrtoutput)
        input_sigmoid = sigmoid(self.input_)
        input_grad = output_grad * input_sigmoid * (1 - input_sigmoid)
        self.input_ = None
        return input_grad


class MSE(Module):
    def __init__(self, reduction: str = "mean") -> None:
        """Mean Squared Loss module

        Args:
            reduction (str, optional): Type of reduction to apply to loss. Defaults to "mean".
                Accepted values ["mean", "sum"].
        """
        super().__init__("MSE")
        self.reduction = reduction
        self.input_ = None
        self.target = None

    def forward(self, *input):
        check_inputs(input, 2)
        input_ = input[0]
        target = input[1]

        if input_.shape != target.shape:
            raise ValueError("Input and target shapes should be same")

        error = (input_ - target) ** 2
        loss = error

        if self.reduction == "sum":
            loss = error.sum()
        elif self.reduction == "mean":
            loss = error.mean()
        
        if not self._training:
            return loss

        self.input_ = input_
        self.target = target
        return autograd_tensor(loss, self, [self.input_, self.target])

    def backward(self, *gradwrtoutput):
        output_grad = get_gradient(gradwrtoutput)
        input_grad = 2 * (self.input_ - self.target)

        if self.reduction == "mean":
            input_grad = input_grad / reduce(lambda a, b: a * b, self.input_.shape)

        self.input_ = None
        self.target = None

        return output_grad * input_grad


class MaxPool2d(Module):
    def __init__(self, kernel_size: Union[int, Tuple],
                 stride: Optional[Union[int, Tuple]] = None,
                 padding: Union[int, Tuple] = 0,
                 dilation: Union[int, Tuple] = 1) -> None:
        """Initialize Max Pooling layer

        Args:
            kernel_size (Union[int, Tuple]): Kernel size
            stride (Optional[Union[int, Tuple]], optional): Stride. Defaults to None.
            padding (Union[int, Tuple], optional): Padding. Defaults to 0.
            dilation (Union[int, Tuple], optional): Dilation. Defaults to 1.
        """
        super().__init__("MaxPool2d")
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size,
                                                                    int) else kernel_size

        if stride is None:
            stride = kernel_size

        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else dilation

    def forward(self, *input):
        check_inputs(input)
        input_ = input[0]
        output = max_pool2d(input_, kernel_size=self.kernel_size,
                            stride=self.stride, padding=self.padding,
                            dilation=self.dilation)
        
        if not self._training:
            return output

        self.input_ = input_
        output = autograd_tensor(output, self, self.input_)
        return output

    def backward(self, *gradwrtoutput):
        output_grad = get_gradient(gradwrtoutput)
        input_grads = []
        N, C, H_in, W_in = self.input_.shape

        # Since channels are independent, we perform it separately
        for ch in range(C):
            input_unfolded = unfold(self.input_[:, ch, :, :].unsqueeze(1),
                                  kernel_size=self.kernel_size,
                                  stride=self.stride, padding=self.padding,
                                  dilation=self.dilation)
            input_grad = zeros_like(input_unfolded)
            input_grad = input_grad.scatter(1,
                                            input_unfolded.argmax(dim=1, keepdims=True),
                                            1)
            input_grad = input_grad * output_grad[:, ch, :, :].reshape((N, 1, -1))
            input_grad = fold(input_grad, output_size=(H_in, W_in),
                              kernel_size=self.kernel_size,
                              stride=self.stride, padding=self.padding,
                              dilation=self.dilation)
            input_grads.append(input_grad)

        self.input_ = None

        return torch.cat(input_grads, dim=1)


Upsampling = TransposeConv2d