import torch
import unittest
import sys
sys.path.append("..")

from parameter import Parameter
from autograd import accumulate_grad, zero_grad


class TestParameter(unittest.TestCase):
    def test_basic(self):
        data = torch.tensor([1.0, 2.0, 3.0])
        parameter = Parameter(data)
        self.assertTrue(torch.equal(parameter.data, data))
        self.assertTrue(torch.equal(parameter.gradient, torch.zeros_like(parameter.gradient)))
        self.assertEqual(parameter.grad_required, True)
    
    def test_accumulate(self):
        data = torch.tensor([1.0, 2.0, 3.0])
        parameter = Parameter(data, grad_required=False)
        self.assertTrue(torch.equal(parameter.gradient, torch.zeros_like(parameter.gradient)))

        grad = torch.tensor([1.0, 0.0, 1.0])

        accumulate_grad(parameter, grad)
        self.assertTrue(torch.equal(parameter.gradient, torch.zeros_like(parameter.gradient)))

        parameter.grad_required = True
        accumulate_grad(parameter, grad)
        self.assertTrue(torch.equal(parameter.gradient, torch.tensor([1.0, 0.0, 1.0])))
        
        zero_grad(parameter)
        self.assertTrue(torch.equal(parameter.gradient, torch.zeros_like(parameter.gradient)))

