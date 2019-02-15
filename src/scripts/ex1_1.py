#!/usr/bin/env python
#  Name:     ex1_1.py
import  numpy as np 
import sys
from osgeo import gdal   
from osgeo.gdalconst import GA_ReadOnly 
import matplotlib.pyplot as plt 
 
def disp(infile,bandnumber):          
    gdal.AllRegister()             
    inDataset = gdal.Open(infile,GA_ReadOnly)     
    cols = inDataset.RasterXSize 
    rows = inDataset.RasterYSize 
    bands = inDataset.RasterCount   

    image = np.zeros((bands,rows,cols))                              
    for b in range(bands):
        band = inDataset.GetRasterBand(b+1)
        image[b,:,:]=band.ReadAsArray(0,0,cols,rows)
    inDataset = None

#  display NIR band    
    band = image[bandnumber-1,:,:]   
    mn = np.amin(band)
    mx = np.amax(band)
    plt.imshow((band-mn)/(mx-mn), cmap='gray')
    plt.show()                        

if __name__ == '__main__':
    infile = sys.argv[1]
    bandnumber = int(sys.argv[2])   
    disp(infile,bandnumber)
