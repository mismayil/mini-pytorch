import torch
from typing import Union

def check_inputs(inputs, length=1):
    if len(inputs) != length:
        raise TypeError(f"Expected {length} inputs, got {len(inputs)}")


def get_gradient(grads):
    if len(grads) > 0:
        check_inputs(grads)
        return grads[0]
    
    return torch.tensor(1.0)

## Implementation of some torch methods
def zeros(shape):
    x = torch.empty(shape)
    x.fill_(0.0)
    return x

def ones(shape):
    x = torch.empty(shape)
    x.fill_(1.0)
    return x

def zeros_like(tensor):
    x = torch.empty(tensor.shape)
    x.fill_(0.0)
    return x

def ones_like(tensor):
    x = torch.empty(tensor.shape)
    x.fill_(1.0)
    return x


def psnr(denoised_image: torch.Tensor, original_image: torch.Tensor,
         max_range: Union[int, float, torch.Tensor] = 1.0,
         device: Union[torch.device, None] = torch.device('cpu')) -> torch.Tensor:
    """
    Computes the PSNR between two images.

    Args:
        denoised_image: the denoised image
        original_image: the original image
        max_range: the maximum value of the image
        device: the device to use

    Returns:
        the PSNR between the two images
    """
    assert denoised_image.shape == original_image.shape and denoised_image.ndim == 4
    return 20 * torch.log10(torch.tensor(max_range, device=device)) - 10 * torch.log10(((denoised_image-original_image) ** 2).mean((1,2,3))).mean()
