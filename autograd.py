from typing import List, Any, Optional, Union
import torch
from functools import partial

try:
    from .utils import get_gradient
except:
    from utils import get_gradient

ATTR_INPUTS = "inputs"
ATTR_OPERATION = "operation"
        
def get_inputs(tensor: torch.Tensor) -> List:
    """Get input tensors used to compute the given tensor.

    Args:
        tensor (torch.Tensor): Tensor

    Returns:
        List: List of input tensors
    """
    if tensor.metadata:
        return tensor.metadata.get(ATTR_INPUTS, [])
    return []

def get_operation(tensor: torch.Tensor) -> Any:
    """Get the operation object used to compute the
    given tensor.

    Args:
        tensor (torch.Tensor): Tensor

    Returns:
        Any: Operation object
    """
    if tensor.metadata:
        return tensor.metadata.get(ATTR_OPERATION)
    return None

def backward(tensor: torch.Tensor, *gradients, **kwargs) -> None:
    """Perform backward propagation from the given tensor.
    It recursively walks the computation graph and 
    calls backward operation on each tensor in the graph.

    Args:
        tensor (torch.Tensor): Tensor
    """
    gradient = get_gradient(gradients)
    operation = get_operation(tensor)

    if operation is not None:
        gradient = operation.backward(gradient)

    inputs = get_inputs(tensor)

    for input_ in inputs:
        if hasattr(input_, "grad_required") and input_.grad_required:
            input_.backward(gradient)
    
    tensor.metadata = {ATTR_OPERATION: None, ATTR_INPUTS: []}
    tensor.backward = None
            
def autograd_tensor(tensor: torch.Tensor, operation: Any = None,
                    inputs: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None) -> torch.Tensor:
    """Construct an automatically differentiable tensor.
    This function registers the inputs and the operation with the given
    tensor to build the computation graph.

    Args:
        tensor (torch.Tensor): Tensor
        operation (Any, optional): Operation used to compute the tensor. Defaults to None.
        inputs (Optional[Union[torch.Tensor, List[torch.Tensor]]], optional): Inputs used for this tensor. Defaults to None.

    Returns:
        torch.Tensor: Autograd enabled tensor 
    """
    if inputs is None:
        inputs = []
    
    if isinstance(inputs, torch.Tensor):
        inputs = [inputs]
    
    tensor.grad_required = True

    tensor.metadata = {ATTR_OPERATION: operation, ATTR_INPUTS: inputs}
    tensor.backward = partial(backward, tensor)

    return tensor

def accumulate_grad(tensor: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    """Accumulate gradient in the tensor.

    Args:
        tensor (torch.Tensor): Tensor
        grad (torch.Tensor): Tensor gradient

    Returns:
        torch.Tensor: Accumulated tensor gradient
    """
    if tensor.grad_required:
        tensor.gradient = tensor.gradient + grad
    return tensor.gradient

def zero_grad(tensor: torch.Tensor) -> None:
    """Zero out tensor gradients

    Args:
        tensor (torch.Tensor): Tensor
    """
    tensor.gradient.zero_()