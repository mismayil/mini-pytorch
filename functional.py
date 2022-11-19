from typing import Optional, Union, Tuple
from math import floor

import torch
from torch.nn.functional import fold, unfold


def linear(input_: torch.Tensor, weight: torch.Tensor,
           bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Apply a linear transformation to the input

    Args:
        input_ (torch.Tensor): Input tensor. Must be of shape (*, IN_DIM)
        weight (torch.Tensor): Weight matrix. Must be of shape (OUT_DIM, IN_DIM)
        bias (Optional[torch.Tensor], optional): Bias vector. Must be of shape (OUT_DIM). Defaults to None.

    Returns:
        torch.Tensor: Transformed input
    """
    output = input_.mm(weight.T)

    if bias is not None:
        output += bias

    return output


def conv2d(input_: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None,
           stride: Union[int, Tuple] = 1, padding: Union[int, Tuple] = 0,
           dilation: Union[int, Tuple] = 1) -> torch.Tensor:
    """Apply a 2d convolution on the input tensor.

    Args:
        input_ (torch.Tensor): Input tensor. Must be of shape (N, C_in, H_in, W_in)
        weight (torch.Tensor): Weight matrix (kernel matrix). Must be of shape (C_out, C_in, H_k, W_k)
        bias (Optional[torch.Tensor], optional): Bias tensor. Must be of shape (C_out,) Defaults to None.
        stride (Union[int, Tuple], optional): Convolution stride. Defaults to 1.
        padding (Union[int, Tuple], optional): Convolution padding. Defaults to 0.
        dilation (Union[int, Tuple], optional): Convolution dilation. Defaults to 1.

    Raises:
        ValueError: When weight and input channels do not match.

    Returns:
        torch.Tensor: Convoluted output tensor.
    """
    stride = (stride, stride) if isinstance(stride, int) else stride
    padding = (padding, padding) if isinstance(padding, int) else padding
    dilation = (dilation, dilation) if isinstance(dilation, int) else dilation

    N, C_in, H_in, W_in = input_.shape
    C_out, C_ker, H_ker, W_ker = weight.shape
    kernel_size = (H_ker, W_ker)

    if C_in != C_ker: raise ValueError(
        "Numbers of channels in the input and kernel are different.")

    H_out = floor((H_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
    W_out = floor((W_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)

    input_ = unfold(input_, kernel_size=kernel_size, padding=padding,
                    stride=stride, dilation=dilation)
    output = input_.transpose(1, 2).matmul(weight.reshape(C_out, -1).T).transpose(1, 2)
    output = output.reshape(N, C_out, H_out, W_out)

    if bias is not None:
        output += bias.reshape(1, -1, 1, 1)

    return output


def conv_transpose2d(input_: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None,
                     stride: Union[int, Tuple] = 1, padding: Union[int, Tuple] = 0,
                     dilation: Union[int, Tuple] = 1) -> torch.Tensor:
    """Apply a 2d transpose convolution on the input tensor.

    Args:
        input_ (torch.Tensor): Input tensor. Must be of shape (N, C_in, H_in, W_in)
        weight (torch.Tensor): Weight matrix (kernel matrix). Must be of shape (C_in, C_out, H_k, W_k)
        bias (Optional[torch.Tensor], optional): Bias data. Must be of shape (C_in, ). Defaults to None.
        stride (Union[int, Tuple], optional): Convolution stride. Defaults to 1.
        padding (Union[int, Tuple], optional): Concolution padding. Defaults to 0.
        dilation (Union[int, Tuple], optional): Convolution dilation. Defaults to 1.

    Raises:
        ValueError: When weight and input channels dont match.

    Returns:
        torch.Tensor: Deconvoluted output tensor.
    """
    stride = (stride, stride) if isinstance(stride, int) else stride
    padding = (padding, padding) if isinstance(padding, int) else padding
    dilation = (dilation, dilation) if isinstance(dilation, int) else dilation

    N, C_in, H_in, W_in = input_.shape
    C_ker, C_out, H_ker, W_ker = weight.shape
    kernel_size = (H_ker, W_ker)

    if C_in != C_ker: raise ValueError(
        "Numbers of channels in the input and kernel are different.")

    input_ = input_.transpose(0, 1).reshape(C_in, -1)
    output = input_.T.matmul(weight.reshape(C_in, -1))
    output = output.reshape(N, H_in * W_in, C_out * kernel_size[0] * kernel_size[1]).transpose(1, 2)

    H_out = (H_in - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_size[0] - 1) + 1
    W_out = (W_in - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_size[1] - 1) + 1

    output = fold(output, (H_out, W_out), kernel_size=kernel_size,
                  dilation=dilation, padding=padding, stride=stride)

    if bias is not None:
        output += bias.reshape(1, -1, 1, 1)

    return output


def relu(input_: torch.Tensor) -> torch.Tensor:
    """Apply relu to input tensor.

    Args:
        input_ (torch.Tensor): Input tensor

    Returns:
        torch.Tensor: Output tensor
    """
    return input_.maximum(torch.tensor(0))


def sigmoid(input_: torch.Tensor) -> torch.Tensor:
    """Apply sigmoid to input tensor.

    Args:
        input_ (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Output tensor.
    """
    return input_.sigmoid()


def max_pool2d(input_: torch.Tensor, kernel_size: Union[int, Tuple],
               stride: Optional[Union[int, Tuple]] = None,
               padding: Union[int, Tuple] = 0, dilation: Union[int, Tuple] = 1) -> torch.Tensor:
    """Apply 2d max pooling to input tensor.

    Args:
        input_ (torch.Tensor): Input tensor. Must be of shape (N, C_in, H_in, W_in).
        kernel_size (Union[int, Tuple]): Kernel size.
        stride (Optional[Union[int, Tuple]], optional): Pooling stride. Defaults to None.
        padding (Union[int, Tuple], optional): Pooling padding. Defaults to 0.
        dilation (Union[int, Tuple], optional): Pooling dilation. Defaults to 1.

    Returns:
        torch.Tensor: Max pooled output tensor.
    """
    kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size,
                                                           int) else kernel_size

    if stride is None:
        stride = kernel_size

    stride = (stride, stride) if isinstance(stride, int) else stride
    padding = (padding, padding) if isinstance(padding, int) else padding
    dilation = (dilation, dilation) if isinstance(dilation, int) else dilation

    N, C, H_in, W_in = input_.shape
    H_out = floor((H_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) /
                  stride[0] + 1)
    W_out = floor((W_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) /
                  stride[1] + 1)

    channel_outputs = []

    for ch in range(C):
        x_ch_unfolded = unfold(input_[:, ch, :, :].unsqueeze(1), kernel_size=kernel_size,
                               stride=stride, dilation=dilation, padding=padding)
        x_ch_max, _ = x_ch_unfolded.max(dim=1, keepdim=True)
        channel_outputs.append(x_ch_max.reshape((N, 1, H_out, W_out)))

    output = torch.cat(channel_outputs, dim=1)

    return output
