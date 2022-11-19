import unittest
import sys
import torch
sys.path.append("..")

from utils import zeros, ones, zeros_like, ones_like


class TestUtils(unittest.TestCase):
    def test_zeros(self):
        self.assertTrue(torch.equal(zeros((2, 4)), torch.zeros((2, 4))))

    def test_ones(self):
        self.assertTrue(torch.equal(ones((2, 4)), torch.ones((2, 4))))

    def test_zeros_like(self):
        x = torch.empty((2, 4))
        self.assertTrue(torch.equal(zeros_like(x), torch.zeros_like(x)))

    def test_ones_like(self):
        x = torch.empty((2, 4))
        self.assertTrue(torch.equal(ones_like(x), torch.ones_like(x)))