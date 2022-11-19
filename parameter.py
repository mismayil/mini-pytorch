import torch

try:
    from .utils import zeros
except:
    from utils import zeros


def Parameter(data: torch.Tensor, grad_required: bool = True) -> torch.Tensor:
    """Create a parameter tensor.
    This simply sets the grad_required attribute
    and initializes the tensor grad to zeros.

    Args:
        data (torch.Tensor): Parameter data.
        grad_required (bool, optional): Whether requires grad. Defaults to True.

    Returns:
        torch.Tensor: Parameter tensor.
    """
    data.grad_required = grad_required
    data.gradient = zeros(data.shape)
    return data
