from torchvision.datasets import MNIST, CIFAR10, CIFAR100

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

'''
MNIST_train_dataset = MNIST(
    root='../data/mnist',
    train=True,
    transform=None,
    download=True
)

MNIST_test_dataset = MNIST(
    root='../data/mnist',
    train=False,
    transform=None,
    download=True
)
'''

CIFAR10_train_dataset = CIFAR10(
    root='../data/cifar10',
    train=True,
    transform=None,
    download=True,
)

CIFAR10_test_dataset = CIFAR10(
    root='../data/cifar10',
    train=False,
    transform=None,
    download=True,
)

CIFAR100_train_dataset = CIFAR100(
    root='../data/cifar100',
    train=True,
    transform=None,
    download=True,
)

CIFAR100_test_dataset = CIFAR100(
    root='../data/cifar100',
    train=False,
    transform=None,
    download=True,
)
