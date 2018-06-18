#!/usr/bin/env python
#Name:  ex3_2.py
import numpy as np
from osgeo import gdal
import sys
from osgeo.gdalconst import GA_ReadOnly

def noisecovar(infile): 
    gdal.AllRegister()                   
    inDataset = gdal.Open(infile,GA_ReadOnly)     
    cols = inDataset.RasterXSize
    rows = inDataset.RasterYSize  
    bands = inDataset.RasterCount  
#  data matrix for difference images       
    D = np.zeros((cols*rows,bands))                           
    for b in range(bands):
        band = inDataset.GetRasterBand(b+1)
        tmp = band.ReadAsArray(0,0,cols,rows)
        D[:,b] = (tmp-np.roll(tmp,1,axis=0)).ravel()      
#  noise covariance matrix
    return np.mat(D).T*np.mat(D)/(2*(rows*cols-1))    
    
if __name__ == '__main__':
    infile = sys.argv[1]
    S_N = noisecovar(infile)
    print 'Noise covariance, file %s'%infile
    print S_N   
