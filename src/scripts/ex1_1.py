#!/usr/bin/env python
#  Name:     ex1_1.py
import  numpy as np 
from osgeo import gdal   
from osgeo.gdalconst import GA_ReadOnly 
import matplotlib.pyplot as plt
 
def main():  
        
    gdal.AllRegister() 
    infile = '../imagery/may0107'
    if infile:                  
        inDataset = gdal.Open(infile,GA_ReadOnly)     
        cols = inDataset.RasterXSize
        rows = inDataset.RasterYSize 
        bands = inDataset.RasterCount   
    else:
        return  

#  BSQ array
    image = np.zeros((bands,rows,cols))                              
    for b in range(bands):
        band = inDataset.GetRasterBand(b+1)
        image[b,:,:]=band.ReadAsArray(0,0,cols,rows)
    inDataset = None

#  display first band    
    band0 = image[2,:,:]   
    mn = np.amin(band0)
    mx = np.amax(band0)
    plt.imshow((band0-mn)/(mx-mn), cmap='gray' )  
    plt.show()                        

if __name__ == '__main__':
    main()    
