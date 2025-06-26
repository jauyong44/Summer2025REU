import torch.nn as nn

ACTIVATION_FUNCTIONS = {
    "ReLU": nn.ReLU,
    "Leaky_ReLU": nn.LeakyReLU,
    "Sigmoid": nn.Sigmoid,
    "Tanh": nn.Tanh,
}