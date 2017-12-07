## This code is written by Davide Albanese, <davide.albanese@gmail.com>.
## (C) 2011 mlpy Developers

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

__all__ = ['MFastHCluster']


import numpy as np
import scipy.cluster.hierarchy as hierarchy
import fastcluster


class MFastHCluster:
    """Memory-saving Hierarchical Cluster (only euclidean distance).

    This method needs O(NP) memory for clustering of N point in R^P.
    """

    def __init__ (self, method='single'):
        """Initialization.

        :Parameters:
          method : string ('single', 'centroid', 'median', 'ward')
            the agglomeration method to be used
        """
        
        self._method = method
        self._Z = None

        self._linkage = False

    def linkage(self, x):
        """Performs hierarchical clustering.

        :Parameters:
          x : 2d array_like object (N, P)
             vector data, N observations in R^P
        """

        self._Z = fastcluster.linkage_vector(X=x, method=self._method, 
            metric='euclidean', extraarg=None)
        
    def Z(self):
        """Returns the hierarchical clustering encoded as a 
        linkage matrix. See `scipy.cluster.hierarchy.linkage`.
        """
        
        return self._Z

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

        if self._Z is None:
            raise ValueError("No linkage computed")

        return hierarchy.fcluster(self._Z, t=t, criterion='distance') - 1
