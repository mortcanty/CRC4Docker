#!/usr/bin/env python
#Name:  ex1_2.py
import numpy as np
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly,GDT_Float32

def main(): 
    gdal.AllRegister()
    infile = '../imagery/may0107' 
                  
    inDataset = gdal.Open(infile,GA_ReadOnly)     
    cols = inDataset.RasterXSize
    rows = inDataset.RasterYSize    
    bands = inDataset.RasterCount
    
#  data matrix
    G = np.zeros((rows*cols,bands)) 
    k = 0                                   
    for b in range(bands):
        band = inDataset.GetRasterBand(b+1)
        tmp = band.ReadAsArray(0,0,cols,rows)\
                              .astype(float).ravel()
        G[:,b] = tmp - np.mean(tmp)
        
#  covariance matrix
    C = np.mat(G).T*np.mat(G)/(cols*rows-1)
    
#  diagonalize    
    lams,U = np.linalg.eigh(C)
     
#  sort
    idx = np.argsort(lams)[::-1]
    lams = lams[idx]
    U = U[:,idx]         
               
#  project
    PCs = np.reshape(np.array(G*U),(rows,cols,bands))   
    
#  write to disk       
    outfile = '../imagery/pca.tif'
    if outfile:
        driver = gdal.GetDriverByName('Gtiff')   
        outDataset = driver.Create(outfile,
                        cols,rows,bands,GDT_Float32)
        projection = inDataset.GetProjection()
        if projection is not None:
            outDataset.SetProjection(projection)        
        for k in range(bands):        
            outBand = outDataset.GetRasterBand(k+1)
            outBand.WriteArray(PCs[:,:,k],0,0) 
            outBand.FlushCache() 
        outDataset = None    
    inDataset = None        
 
if __name__ == '__main__':
    main()    