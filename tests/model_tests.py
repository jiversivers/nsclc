import unittest
import inspect
import sys

from my_modules.custom_models import *

sys.path.append('C:\\Users\\jdivers\\PycharmProjects\\NSCLC_Classification')

class BinaryClassifierTests(unittest.TestCase):
    def setUp(self):
        self.model = None

    def test_forward_with_images(self):
        self.model = self.get_model()
        print(self.model)

        # Normal case of multi-modal image data
        x = torch.randn(32, 4, 512, 512)
        mod = self.model(x.shape[1:])
        print(mod)
        out = mod(x)
        self.assertTrue((out.shape == (32, 1) and out.dtype == torch.float32))

        # Batch size of 1 with multi-modal image data
        x = torch.randn(1, 4, 512, 512)
        mod = self.model(x.shape[1:])
        out = mod(x)
        self.assertTrue((out.shape == (1, 1) and out.dtype == torch.float32))

        # Batched mono-modal image data
        x = torch.randn(32, 1, 512, 512)
        mod = self.model(x.shape[1:])
        out = mod(x)
        self.assertTrue((out.shape == (32, 1) and out.dtype == torch.float32))

        # Batch size of 1 mono-modal
        x = torch.randn(1, 1, 512, 512)
        mod = self.model(x.shape[1:])
        out = mod(x)
        self.assertTrue((out.shape == (1, 1) and out.dtype == torch.float32))

    def test_forward_sizes_with_dists(self):

        self.model = self.get_model()

        # Normal case of hist data
        x = torch.randn(32, 4, 25)
        mod = self.model(x.shape[1:])
        out = mod(x)
        self.assertTrue((out.shape == (32, 1) and out.dtype == torch.float32))

        # Batch size of 1 with multi-modal image data
        x = torch.randn(1, 4, 25)
        mod = self.model(x.shape[1:])
        out = mod(x)
        self.assertTrue((out.shape == (1, 1) and out.dtype == torch.float32))

        # Batched mono-modal image data
        x = torch.randn(32, 25)
        mod = self.model(x.shape[1:])
        out = mod(x)
        self.assertTrue((out.shape == (32, 1) and out.dtype == torch.float32))

        # Batch size of 1 mono-modal
        x = torch.randn(1, 25)
        mod = self.model(x.shape[1:])
        out = mod(x)
        self.assertTrue((out.shape == (1, 1) and out.dtype == torch.float32))

    def get_model(self):
        raise NotImplementedError('Subclasses must implement get_model')


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
