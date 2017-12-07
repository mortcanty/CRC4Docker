## This code is written by Davide Albanese, <albanese@fbk.eu>.
## (C) 2010 mlpy Developers.

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

import numpy as np

__all__ = ['uwt_align_h2', 'uwt_align_d4']


def uwt_align_h2(X, inverse=False):
    """UWT h2 coefficients aligment.

    If inverse = True performs the misalignment
    for a correct reconstruction.
    """
    
    J = X.shape[0] / 2
    shifts = np.asarray([2**j for j in range(J)])
    
    if not inverse:
        shifts *= -1

    for j in range(J):
        X[j] = np.roll(X[j], shifts[j])
        X[j+J] = np.roll(X[j+J], shifts[j])
    

def uwt_align_d4(X, inverse=False):
    """UWT d4 coefficients aligment.

    If inverse = True performs the misalignment
    for a correct reconstruction.
    """
    J = X.shape[0] / 2
    w_shifts = np.asarray([(3 * 2**j) - 1 for j in range(J)])
    v_shifts = np.asarray([1] + [(2**(j+1) - 1) for j in range(1, J)])
    
    if not inverse:
        w_shifts *= -1
        v_shifts *= -1

    for j in range(J):
        X[j] = np.roll(X[j], w_shifts[j])
        X[j+J] = np.roll(X[j+J], v_shifts[j])
