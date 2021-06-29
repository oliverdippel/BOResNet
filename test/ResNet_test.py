import unittest
import torch
import numpy as np
from ResNet import ResNet

#np.load('data\k49-test-imgs.npz')


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

    #     resnet = ResNet(
    #         img_size=(40, 40),
    #         architecture=((1, 64), (64, 64, 64), (64, 128, 128)),
    #         no_classes=10)
    #
    #     self.assertEqual(resnet(torch.ones((1, 1, 40, 40))).shape,
    #                      torch.Size([1, 10]))
    #
    # def test_scaling_architechture(self):
    #     """check different architechtures incl. irregular sized residblocks
    #     and differently sized channels work as well."""
    #     # using a different net to the ones used in
    #     # test_architechture_scaling_data (more residblocks skip=2 & halved
    #     # no. of channels.
    #     resnet = ResNet(
    #         img_size=(28, 28),
    #         architecture=((1, 32), (32, 32, 32), (32, 32, 32), (32, 64, 64)),
    #         no_classes=10)
    #
    #     self.assertEqual(resnet(torch.ones((1, 1, 28, 28))).shape,
    #                      torch.Size([1, 10]))
    #
    #     # dotted connection (32, 64, 64) implies 1x1 conv!
    #     # here with a skip of 3
    #     resnet = ResNet(
    #         img_size=(28, 28),
    #         architecture=((1, 32), (32, 32, 32), (32, 32, 32, 32),
    #                       (32, 64, 64)),
    #         no_classes=10)
    #
    #     self.assertEqual(resnet(torch.ones((1, 1, 28, 28))).shape,
    #                      torch.Size([1, 10]))
    #
    #     # dotted connection (32, 64, 64, 64) implies 1x1 conv!
    #     # here with a skip of 3
    #     resnet = ResNet(
    #         img_size=(28, 28),
    #         architecture=((1, 32), (32, 32, 32), (32, 32, 32),
    #                       (32, 64, 64, 64)),
    #         no_classes=10)
    #
    #     self.assertEqual(resnet(torch.ones((1, 1, 28, 28))).shape,
    #                      torch.Size([1, 10]))


if __name__ == '__main__':
    unittest.main(exit=False)