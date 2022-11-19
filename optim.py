from typing import List

try:
    from .parameter import Parameter
    from .autograd import zero_grad
except:
    from parameter import Parameter
    from autograd import zero_grad


class Optimizer(object):
    def __init__(self, parameters: List[Parameter]) -> None:
        self.parameters = parameters

    def zero_grad(self):
        raise NotImplementedError
    
    def step(self):
        raise NotImplementedError

    def load_parameters(self, parameters):
        self.parameters = parameters


class SGD(Optimizer):
    def __init__(self, parameters: List[Parameter], lr: float = 0.01) -> None:
        super().__init__(parameters)
        self.lr = lr
    
    def zero_grad(self):
        for parameter in self.parameters:
            zero_grad(parameter)
        
    def step(self):
        for parameter in self.parameters:
            parameter.data = parameter.data - self.lr * parameter.gradient