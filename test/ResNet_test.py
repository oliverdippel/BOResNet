import unittest
import torch
from ResNet import ResNet


class TestResNet(unittest.TestCase):

    def test_architechture_scaling_data(self):
        """
        check that the scaling of the image does not break the net's
        definition (linear == fully connected layer is critical in thin
        regard)
        """
        resnet = ResNet(img_size=(28, 28))

        self.assertEqual(resnet(torch.ones((1, 1, 28, 28))).shape,
                         torch.Size([1, 10]))


if __name__ == '__main__':
    unittest.main(exit=False)
