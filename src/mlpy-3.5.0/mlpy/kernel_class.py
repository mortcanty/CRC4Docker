## This code is written by Davide Albanese, <albanese@fbk.eu>.
## (C) 2011 mlpy Developers.

## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.

## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.

## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

__all__ = ["Kernel", "KernelLinear", "KernelPolynomial", "KernelGaussian",
           "KernelExponential", "KernelSigmoid"]


import sys
if sys.version >= '3':
    from . import kernel
else:
    import kernel

class Kernel:
    """Base class for kernels.
    """

    pass

class KernelLinear(Kernel):
    """Linear kernel, t_i' x_j.
    """
    def __init__(self):
        pass
    def kernel(self, t, x):
        return kernel.kernel_linear(t, x)

class KernelPolynomial(Kernel):
    """Polynomial kernel, (gamma t_i' x_j + b)^d.
    """
    def __init__(self, gamma=1.0, b=1.0, d=2.0):
        self.gamma = gamma
        self.b = b
        self.d = d
    def kernel(self, t, x):
        return kernel.kernel_polynomial(t, x,
                   self.gamma, self.b, self.d)

class KernelGaussian(Kernel):
    """Gaussian kernel, exp(-||t_i - x_j||^2 / 2 * sigma^2).
    """
    def __init__(self, sigma=1.0):
        self.sigma = sigma
    def kernel(self, t, x):
        return kernel.kernel_gaussian(t, x,
                   self.sigma)

class KernelExponential(Kernel):
    """Exponential kernel, exp(-||t_i - x_j|| / 2 * sigma^2).
    """
    def __init__(self, sigma=1.0):
        self.sigma = sigma
    def kernel(self, t, x):
        return kernel.kernel_exponential(t, x,
                   self.sigma)

class KernelSigmoid(Kernel):
    """Sigmoid kernel, tanh(gamma t_i' x_j + b).
    """
    def __init__(self, gamma=1.0, b=1.0):
        self.gamma = gamma
        self.b = b
    def kernel(self, t, x):
        return kernel.kernel_sigmoid(t, x,
            self.gamma, self.b)
