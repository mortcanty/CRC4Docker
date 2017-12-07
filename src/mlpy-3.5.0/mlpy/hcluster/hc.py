## This code is written by Davide Albanese, <albanese@fbk.eu>.
## (C) 2009 Fondazione Bruno Kessler - Via Santa Croce 77, 38100 Trento, ITALY.

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

__all__ = ['HCluster']


import numpy as np

import sys
if sys.version >= '3':
    from . import chc
else:
    import chc


class HCluster:
    """Hierarchical Cluster.
    """

    METHODS = {
        'ward': 1,
        'single': 2,
        'complete': 3,
        'average': 4,
        'mcquitty': 5,
        'median': 6,
        'centroid': 7
        }

    def __init__ (self, method='complete'):
        """Initialization.

        :Parameters:
          method : string ('ward', 'single', 'complete', 'average', 'mcquitty', 'median', 'centroid')
            the agglomeration method to be used
        """
        
        self.method = self.METHODS[method]
        
        self._ia = None
        self._ib = None
        self._order = None
        self._height = None

        self._linkage = False

    def linkage(self, y):
        """Performs hierarchical clustering on the condensed
        distance matrix y.

        :Parameters:
          y : 1d array_like object
            condensed distance matrix y. y must be a C(n, 2) sized 
            vector where n is the number of original observations 
            paired in the distance matrix. 
        """

        y_a = np.asarray(y)
        l = y.shape[0]

        if l == 0:
            raise ValueError("invalid condensed distance matrix y")

        n = int(np.ceil(np.sqrt(l * 2)))

        if (n*(n-1)/2) != l:
            raise ValueError("invalid condensed distance matrix y")

        self._ia, self._ib, self._order, self._height = \
            chc.linkage(n, y, self.method)
        
        self._linkage = True

    def cut(self, t):
        """Cuts the tree into several groups by specifying the cut
        height.
        
        :Parameters:
          t : float
            the threshold to apply when forming flat clusters
         
        :Returns:
          clust : 1d numpy array
            group memberships. Groups are in 0, ..., N-1.
        """

        if self._linkage == False:
            raise ValueError("No linkage computed")

        return chc.cut(self._ia, self._ib, self._height, t) - 1
