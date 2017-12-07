## This code is written by Davide Albanese, <davide.albanese@gmail.com>.
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

__all__ = ['pad']

import numpy as np


def next_p2(n):
    """Returns the smallest integer, greater than n
    (n positive and >= 1) which can be obtained as power of 2.
    """
    
    if n < 1:
        raise ValueError("n must be >= 1")

    v = 2
    while v <= n:
        v = v * 2

    return v


def pad(x, method='reflection'):
    """Pad to bring the total length N up to the next-higher 
    power of two.

    :Parameters:
       x : 1d array_like object
          data
       method : string ('reflection', 'periodic', 'zeros')
          method
          
    :Returns:
       xp, orig : 1d numpy array, 1d numpy array bool
          padded version of `x` and a boolean array with
          value True where xp contains the original data
    """

    x_arr = np.asarray(x)

    if not method in ['reflection', 'periodic', 'zeros']:
        raise ValueError('method %s not available' % method)
    
    diff = next_p2(x_arr.shape[0]) - x_arr.shape[0]
    ldiff = int(diff / 2)
    rdiff = diff - ldiff

    if method == 'reflection':
        left_x = x_arr[:ldiff][::-1]
        right_x = x_arr[-rdiff:][::-1]         
    elif method == 'periodic':
        left_x = x_arr[:ldiff]
        right_x = x_arr[-rdiff:]
    elif method == 'zeros':
        left_x = np.zeros(ldiff, dtype=x_arr.dtype)
        right_x = np.zeros(rdiff, dtype=x_arr.dtype)
        
    xp = np.concatenate((left_x, x_arr, right_x))
    orig = np.ones(x_arr.shape[0] + diff, dtype=np.bool)
    orig[:ldiff] = False
    orig[-rdiff:] = False
   
    return xp, orig
