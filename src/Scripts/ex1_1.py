#!/usr/bin/env python
#  Name:     ex1_1.py
import  numpy as np 
from osgeo import gdal   
from osgeo.gdalconst import GA_ReadOnly 
import matplotlib.pyplot as plt
 
def main():  
        
    gdal.AllRegister() 
    infile = '../Images/may0107'
    if infile:                  
        inDataset = gdal.Open(infile,GA_ReadOnly)     
        cols = inDataset.RasterXSize
        rows = inDataset.RasterYSize    
    else:
        return  

    pos = [1,2,3]

#  BSQ array
    image = np.zeros((len(pos),rows,cols)) 
    k = 0                                   
    for b in pos:
        band = inDataset.GetRasterBand(b)
        image[k,:,:]=band.ReadAsArray(0,0,cols,rows)\
                                        .astype(float)
        k += 1
    inDataset = None

#  display first band    
    band0 = image[0,:,:]   
    mn = np.amin(band0)
    mx = np.amax(band0)
    plt.imshow((band0-mn)/(mx-mn), cmap='gray' )  
    plt.show()                        

if __name__ == '__main__':
    main()    
