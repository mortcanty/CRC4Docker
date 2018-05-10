#!/usr/bin/env python
#Name:  ex5_1.py
import numpy as np
import os, sys, getopt
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly,GDT_Float32
import cv2 as cv 

def main(): 
    options,args = getopt.getopt(sys.argv[1:],'b:a:')
    b = 1
    algorithm = 1
    for option, value in options: 
        if option == '-b':
            b = eval(value)          
        elif option == '-a':
            algorithm = eval(value)
    gdal.AllRegister()
    infile = args[0]      
    path = os.path.dirname(infile)
    basename = os.path.basename(infile)
    root, ext = os.path.splitext(basename)        
    inDataset = gdal.Open(infile,GA_ReadOnly)     
    cols = inDataset.RasterXSize
    rows = inDataset.RasterYSize     
    rasterBand = inDataset.GetRasterBand(b)
    band = rasterBand.ReadAsArray(0,0,cols,rows) \
                                 .astype(np.uint8) 
    if algorithm==1:      
#      corner detection, window size 7x7
        result = cv.cornerMinEigenVal(band, 7) 
        outfile = path+'/'+root+'_corner'+ext  
    else:
#      edge detection, window size 7x7
        result = cv.Canny(band,50,150)  
        outfile = path+'/'+root+'_canny'+ext           
#  write to disk       
    driver = inDataset.GetDriver()    
    outDataset = driver.Create(outfile,
                   cols,rows,1,GDT_Float32)         
    outBand = outDataset.GetRasterBand(1)
    outBand.WriteArray(result,0,0) 
    outBand.FlushCache() 
    outDataset = None; inDataset = None     
    print 'result written to %s'%outfile    
if __name__ == '__main__':
    main()    