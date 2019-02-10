#!/usr/bin/env python
#Name:  ex3_1.py
import numpy as np
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly

def main(): 
    gdal.AllRegister()
    infile = 'imagery/LE7_20010626'
    if infile:                  
        inDataset = gdal.Open(infile,GA_ReadOnly)     
        cols = inDataset.RasterXSize
        rows = inDataset.RasterYSize    
        bands = inDataset.RasterCount
    else:
        return
#  transposed data matrix
    m = rows*cols
    G = np.zeros((bands,m))                                  
    for b in range(bands):
        band = inDataset.GetRasterBand(b+1)
        tmp = band.ReadAsArray(0,0,cols,rows)\
                              .astype(float).ravel()
        G[b,:] = tmp - np.mean(tmp) 
    G = np.mat(G)           
#  covariance matrix
    S = G*G.T/(m-1)   
#  diagonalize and sort eigenvectors  
    lamda,W = np.linalg.eigh(S)
    idx = np.argsort(lamda)[::-1]
    lamda = lamda[idx]
    W = W[:,idx]                    
#  get principal components and reconstruct
    r = 2
    Y = W.T*G    
    G_r = W[:,:r]*Y[:r,:]
#  reconstruction error covariance matrix
    print  ((G-G_r)*(G-G_r).T/(m-1))[:3,:3]
#  Equation (3.45)       
    print (W[:,r:]*np.diag(lamda[r:])*W[:,r:].T)[:3,:3]                       
    inDataset = None        
 
if __name__ == '__main__':
    main()    
