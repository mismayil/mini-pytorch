import unittest
import sys
sys.path.append("..")
import torch

from module import Module


def setup_modules():
    module = Module("Module1")
    submodule1 = Module("Submodule1")
    submodule2 = Module("Submodule2")
    subsubmodule1 = Module("Subsubmodule1")
    subsubmodule2 = Module("Subsubmodule2")

    subsubmodule1.register_parameter("weight", torch.tensor(1.0))
    subsubmodule1.register_parameter("bias", torch.tensor(0.0))
    subsubmodule2.register_parameter("weight", torch.tensor(3.0))
    subsubmodule2.register_parameter("bias", torch.tensor(0.0))

    submodule1.register_module(subsubmodule1)
    submodule1.register_module(subsubmodule2)

    submodule2.register_parameter("weight", torch.tensor(5.0))
    submodule2.register_parameter("bias", torch.tensor(7.0))

    module.register_module(submodule1)
    module.register_module(submodule2)

    return module, submodule1, submodule2, subsubmodule1, subsubmodule2

class TestModule(unittest.TestCase):
    def test_module_parameters(self):
        module, submodule1, submodule2, subsubmodule1, subsubmodule2 = setup_modules()
        named_parameters = subsubmodule1.named_parameters()
        name_set = set([name for name, _ in named_parameters])
        self.assertEqual(len(name_set), 2)
        self.assertEqual(set(["weight", "bias"]), name_set)

        for name, parameter in named_parameters:
            if name == "weight":
                self.assertTrue(torch.allclose(parameter, torch.tensor(1.0)))
            else:
                self.assertTrue(torch.allclose(parameter, torch.tensor(0.0)))

        named_parameters = module.named_parameters()
        name_set = set([name for name, _ in named_parameters])
        self.assertEqual(len(name_set), 6)
        expected_name_set = set(["Submodule1#0.Subsubmodule1#0.weight",
                                 "Submodule1#0.Subsubmodule1#0.bias",
                                 "Submodule1#0.Subsubmodule2#1.weight",
                                 "Submodule1#0.Subsubmodule2#1.bias",
                                 "Submodule2#1.weight",
                                 "Submodule2#1.bias"])
        self.assertEqual(expected_name_set, name_set)

        for name, parameter in named_parameters:
            if name == "Submodule1#0.Subsubmodule1#0.weight":
                self.assertTrue(torch.allclose(parameter, torch.tensor(1.0)))
            elif name == "Submodule1#0.Subsubmodule1#0.bias":
                self.assertTrue(torch.allclose(parameter, torch.tensor(0.0)))
            elif name == "Submodule1#0.Subsubmodule2#1.weight":
                self.assertTrue(torch.allclose(parameter, torch.tensor(3.0)))
            elif name == "Submodule1#0.Subsubmodule2#1.bias":
                self.assertTrue(torch.allclose(parameter, torch.tensor(0.0)))
            elif name == "Submodule2#1.weight":
                self.assertTrue(torch.allclose(parameter, torch.tensor(5.0)))
            elif name == "Submodule2#1.bias":
                self.assertTrue(torch.allclose(parameter, torch.tensor(7.0)))
    
    def test_load_state_dict(self):
        module, submodule1, submodule2, subsubmodule1, subsubmodule2 = setup_modules()
        state_dict = module.state_dict()

        for name, parameter in state_dict.items():
            state_dict[name] = parameter + 1
        print(state_dict)
        module.load_state_dict(state_dict)
        named_parameters = module.named_parameters()

        for name, parameter in named_parameters:
            if name == "Submodule1#0.Subsubmodule1#0.weight":
                print(parameter)
                self.assertTrue(torch.allclose(parameter, torch.tensor(2.0)))
            elif name == "Submodule1#0.Subsubmodule1#0.bias":
                self.assertTrue(torch.allclose(parameter, torch.tensor(1.0)))
            elif name == "Submodule1#0.Subsubmodule2#1.weight":
                self.assertTrue(torch.allclose(parameter, torch.tensor(4.0)))
            elif name == "Submodule1#0.Subsubmodule2#1.bias":
                self.assertTrue(torch.allclose(parameter, torch.tensor(1.0)))
            elif name == "Submodule2#1.weight":
                self.assertTrue(torch.allclose(parameter, torch.tensor(6.0)))
            elif name == "Submodule2#1.bias":
                self.assertTrue(torch.allclose(parameter, torch.tensor(8.0)))