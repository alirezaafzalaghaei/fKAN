import torch
import torch.nn as nn
import torch.nn.functional as F

from .jacobi_polynomials import jacobi_polynomial


class FractionalJacobiNeuralBlock(nn.Module):
    """
    Fractional Jacobi Neural Block layer for PyTorch.

    This layer computes a custom transformation using the Jacobi polynomial.

    Attributes:
        degree (int): Degree of the Jacobi polynomial.
    """

    def __init__(self, degree):
        """
        Initialize the Fractional Jacobi Neural Block.

        Args:
            degree (int): Degree of the Jacobi polynomial.
        """
        super(FractionalJacobiNeuralBlock, self).__init__()
        self.degree = degree
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, inputs):
        """
        Forward pass of the layer.

        Args:
            inputs (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying the Jacobi polynomial transformation.
        """
        normalized_alpha = F.elu(self.alpha, 1)
        normalized_beta = F.elu(self.beta, 1)
        normalized_gamma = torch.sigmoid(self.gamma)
        normalized_inputs = torch.sigmoid(inputs)

        return jacobi_polynomial(
            normalized_inputs,
            self.degree,
            normalized_alpha,
            normalized_beta,
            normalized_gamma,
            0,
            1,
        )
