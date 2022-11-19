import unittest
import torch
import sys
sys.path.append("..")

from optim import SGD
from modules import Linear, ReLU, Sequential, MSE, Sigmoid
from parameter import Parameter

class TestOptim(unittest.TestCase):
    def test_sgd(self):
        x = torch.rand((3, 2), requires_grad=True)
        y = torch.rand((3, 2))
        torch_linear1 = torch.nn.Linear(2, 4)
        torch_linear2 = torch.nn.Linear(4, 2)
        torch_model = torch.nn.Sequential(torch_linear1, torch.nn.Sigmoid(), torch_linear2, torch.nn.ReLU())
        linear1 = Linear(2, 4)
        linear2 = Linear(4, 2)
        linear1.weight = Parameter(torch_linear1.weight.data)
        linear1.bias = Parameter(torch_linear1.bias.data)
        linear2.weight = Parameter(torch_linear2.weight.data)
        linear2.bias = Parameter(torch_linear2.bias.data)
        model = Sequential(linear1, Sigmoid(), linear2, ReLU())

        optimizer = SGD(model.parameters(), lr=0.001)
        optimizer.zero_grad()
        out = model(x)
        criterion = MSE()
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        torch_optimizer = torch.optim.SGD(torch_model.parameters(), lr=0.001)
        torch_optimizer.zero_grad()
        torch_out = torch_model(x)
        torch_criterion = torch.nn.MSELoss()
        torch_loss = torch_criterion(torch_out, y)
        torch_loss.backward()
        torch_optimizer.step()

        self.assertEqual(torch_linear1.weight.grad.shape, linear1.weight.gradient.shape)
        self.assertEqual(torch_linear1.bias.grad.shape, linear1.bias.gradient.shape)
        self.assertTrue(torch.allclose(torch_linear1.weight.grad, linear1.weight.gradient))
        self.assertTrue(torch.allclose(torch_linear1.bias.grad, linear1.bias.gradient))