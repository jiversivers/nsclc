import unittest
import inspect
import sys
from abc import ABC, abstractmethod

from my_modules.custom_models import *

sys.path.append('C:\\Users\\jdivers\\PycharmProjects\\NSCLC_Classification')


class BinaryClassifierTests(unittest.TestCase):
    def setUp(self):
        self.model = self.get_model()

    def test_forward_with_images(self):
        # Normal case of multi-modal image data
        x = torch.randn(32, 4, 512, 512)
        print(x.shape[1:])
        mod = self.model(tuple(x.shape[1:]))
        out = mod(x)
        self.assertTrue((out.shape == (32, 1) and out.dtype == torch.float32))

        # Batch size of 1 with multi-modal image data
        x = torch.randn(1, 4, 512, 512)
        mod = self.model(tuple(x.shape[1:]))
        out = mod(x)
        self.assertTrue((out.shape == (1, 1) and out.dtype == torch.float32))

        # Batched mono-modal image data
        x = torch.randn(32, 1, 512, 512)
        mod = self.model(tuple(x.shape[1:]))
        out = mod(x)
        self.assertTrue((out.shape == (32, 1) and out.dtype == torch.float32))

        # Batch size of 1 mono-modal
        x = torch.randn(1, 1, 512, 512)
        mod = self.model(tuple(x.shape[1:]))
        out = mod(x)
        self.assertTrue((out.shape == (1, 1) and out.dtype == torch.float32))

    def test_forward_sizes_with_dists(self):
        # Normal case of hist data
        x = torch.randn(32, 4, 25)
        mod = self.model(tuple(x.shape[1:]))
        print(x.shape[1:])
        out = mod(x)
        self.assertTrue((out.shape == (32, 1) and out.dtype == torch.float32))

        # Batch size of 1 with multi-modal image data
        x = torch.randn(1, 4, 25)
        mod = self.model(tuple(x.shape[1:]))
        out = mod(x)
        self.assertTrue((out.shape == (1, 1) and out.dtype == torch.float32))

        # Batched mono-modal image data
        x = torch.randn(32, 25)
        mod = self.model(tuple(x.shape[1:]))
        out = mod(x)
        self.assertTrue((out.shape == (32, 1) and out.dtype == torch.float32))

        # Batch size of 1 mono-modal
        x = torch.randn(1, 25)
        mod = self.model(tuple(x.shape[1:]))
        out = mod(x)
        self.assertTrue((out.shape == (1, 1) and out.dtype == torch.float32))

    @abstractmethod
    def get_model(self):
        pass

class TestEachModel(BinaryClassifierTests, ABC):
    pass


# Get models from module
models = inspect.getmembers(classifier_models, inspect.isclass)
for m in models:
    class TestEachModel(BinaryClassifierTests):
        def get_model(self):
            return m[1]


    TestEachModel.__name__ = f'Test {m[0]}'

    globals()[TestEachModel.__name__] = TestEachModel

if __name__ == '__main__':
    unittest.main()
