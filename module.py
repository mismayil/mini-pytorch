from typing import Optional, Any, List, Tuple

import torch

PARAMETER_DELIMITER = "."
MODULE_DELIMITER = "#"

try:
    from .parameter import Parameter
    from .autograd import zero_grad
except:
    from parameter import Parameter
    from autograd import zero_grad


class Module(object):
    def __init__(self, name: Optional[str] = None) -> None:
        """Initialize a module.

        Args:
            name (Optional[str], optional): Module name. Defaults to None.
        """
        self.name = name
        self._parameters = {}
        self._training = True
        self._submodules = []

    def register_module(self, module: "Module") -> None:
        """Register a submodule.

        Args:
            module (Module): Submodule
        """
        self._submodules.append(module)

    def modules(self):
        return self._submodules

    def register_parameter(self, name: str, parameter: Optional[Parameter] = None) -> None:
        """Register a parameter with this module

        Args:
            name (str): Parameter name
            parameter (Optional[Parameter], optional): Parameter to register. Defaults to None.
        """
        self._parameters[name] = parameter

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError
    
    def parameters(self, recurse: bool = True) -> List[Parameter]:
        """Get a list of parameters.
        If recures is true, This method recursively finds all
        parameters of the nested modules.

        Args:
            recurse (bool, optional): Whether to recurse module tree. Defaults to True.

        Returns:
            List[Parameter]: List of parameters
        """
        return [parameter for _, parameter in self.named_parameters(recurse=recurse)]

    def named_parameters(self, prefix: str = "", recurse: bool = True) -> List[Tuple[str, Parameter]]:
        """Returns a list of parameters with their names.
        If recurse is true, also retrieves all the parameters of submodules.
        We can also attach a prefix to a parameter name. This is used to construct a unique
        parameter name for each parameter so that we can easily save the model state.
        For example if the model arch is Sequential(Linear(), Sequential(Linear, ReLU()))
        then the following names will be assigned to the linear module parameters:
        "Linear#0.weight", "Linear#0.bias", "Sequential#1.Linear#0.weight", "Sequential#1.Linear#0.bias".
        Here numbers after the # sign indicate the index of the module in its parent submodules list.
        This specific structure is used in saving and loading parameters.

        Args:
            prefix (str, optional): Prefix to attach to parameter name. Defaults to "".
            recurse (bool, optional): Whether to recurse module tree. Defaults to True.

        Returns:
            List[Tuple[str, Parameter]]: List of parameter name and parameter pairs.
        """
        if recurse and len(self.modules()) > 0:
            named_parameters = []

            for i, module in enumerate(self.modules()):
                named_parameters.extend(module.named_parameters(prefix=f"{prefix}{module.name}{MODULE_DELIMITER}{i}{PARAMETER_DELIMITER}",
                                                                recurse=recurse))
            
            return named_parameters

        return [(f"{prefix}{name}", parameter) for name, parameter in self._parameters.items() if parameter is not None]

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args)
    
    def train(self) -> None:
        """
        Set mddule and its submodules in training mode
        """
        self._training = True

        for module in self.modules():
            module.train()
    
    def eval(self) -> None:
        """
        Set module and its submodules in eval mode
        """
        self._training = False

        for module in self.modules():
            module.eval()
    
    def zero_grad(self) -> None:
        """
        Zero out all module parameters
        """
        for parameter in self.parameters():
            zero_grad(parameter)
        
    def state_dict(self) -> dict:
        """Return a dictionary of the module state.
        This is simple dictionary of all module parameters.

        Returns:
            dict: Module state
        """
        named_parameters = self.named_parameters()
        return {name: parameter for name, parameter in named_parameters}
    
    def load_parameter(self, name: str, parameter: torch.Tensor) -> None:
        """Load parameter in the module.
        This method recursively locates the parameter in the module's
        submodules based on the canonically constructed parameter name.

        Args:
            name (str): Parameter name
            parameter (torch.Tensor): Parameter tensor

        Raises:
            ValueError: If the module name is not found in the submodules.
        """
        name_parts = name.split(PARAMETER_DELIMITER)

        if len(name_parts) == 1:
            parameter = Parameter(parameter)
            setattr(self, name, parameter)
            self.register_parameter(name, parameter)
            return
        
        module_part = name_parts[0]
        module_name, module_num = module_part.split(MODULE_DELIMITER)
        module = self.modules()[int(module_num)]

        if module.name != module_name:
            raise ValueError(f"Invalid state dict. {module_name} not found in submodules")

        module.load_parameter(name[name.find(PARAMETER_DELIMITER)+1:], parameter)

    def load_state_dict(self, state: dict) -> "Module":
        """Load module state.

        Args:
            state (dict): Module state

        Returns:
            Module: Module loaded with the state
        """
        for name, parameter in state.items():
            self.load_parameter(name, parameter)
        
        return self
    
    def to(self, device: torch.device) -> "Module":
        """Cast module to a torch device.
        This simply amounts to casting all of module's
        parameters to the device.

        Args:
            device (torch.device): Torch device

        Returns:
            Module: Module loaded on the device
        """
        if len(self.modules()) > 0:
            for module in self.modules():
                module.to(device)
            return self
        
        for name, parameter in self.named_parameters():
            parameter.data = parameter.data.to(device)
            parameter.gradient = parameter.gradient.to(device)
            setattr(self, name, parameter)
            self.register_parameter(name, parameter)

        return self
    
    def param(self):
        return [(parameter, parameter.gradient) for parameter in self.parameters()]
