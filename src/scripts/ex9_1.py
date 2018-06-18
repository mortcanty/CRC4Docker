#!/usr/bin/env python
#Name:  ex9_1.py

import sys
import numpy as np
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly
import matplotlib.pyplot as plt
import auxil.auxil1 as auxil
import  em
    
def main(): 
#  read in bands 4
    infile1 = sys.argv[1]
    infile2 = sys.argv[2] 
    inDataset = gdal.Open(infile1,GA_ReadOnly)     
    cols = inDataset.RasterXSize
    rows = inDataset.RasterYSize    
    band = inDataset.GetRasterBand(4)  
    G1 = band.ReadAsArray(0,0,cols,rows).flatten()    
    inDataset = gdal.Open(infile2,GA_ReadOnly)       
    band = inDataset.GetRasterBand(4)  
    G2 = band.ReadAsArray(0,0,cols,rows).flatten()
#  centered data matrix    
    G = np.zeros((rows*cols,2))
    G[:,0] = G1-np.mean(G1)
    G[:,1] = G2-np.mean(G2)
#  initial PCA
    cpm = auxil.Cpm(2) 
    cpm.update(G)  
    eivs,w = np.linalg.eigh(cpm.covariance())
    eivs = eivs[::-1]
    w = w[:,::-1]                    
    pcs = G*w       
    plt.plot([-1,1],[-np.abs(w[0,1]/w[0,0]),
                     np.abs(w[0,1]/w[0,0])])
#  iterated PCA    
    itr = 0
    while itr<5:
        sigma = np.sqrt(eivs[1])
        U = np.random.rand(2,rows*cols)
#      cluster the second PC
        unfrozen=np.where(np.abs(pcs[:,1]) >= sigma)[0]
        frozen=np.where( np.abs(pcs[:,1]) < sigma)[0]
        U[0,frozen] = 1.0
        U[1,frozen] = 0.0
        for j in range(2):
            U[j,:]=U[j,:]/np.sum(U,0)
        U=em.em(G,U,0,0,rows,cols,unfrozen=unfrozen)[0]   
#      re-sample the weighted covariance matrix 
        cpm.update(G,U[0,:])
        cov = cpm.covariance()
        eivs,w = np.linalg.eigh(cov)
        eivs = eivs[::-1]
        w = w[:,::-1]    
#      weighted PCs                
        pcs = G*w 
#      plot the first principal axis   
        plt.plot([-1,1],[-np.abs(w[0,1]/w[0,0]),
                np.abs(w[0,1]/w[0,0])],dashes=[4,4])
        itr += 1
    plt.show()
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(np.reshape(U[1,:],(rows,cols)),cmap='gray')  
    plt.show()     
                      
if __name__ == '__main__':
    main()    