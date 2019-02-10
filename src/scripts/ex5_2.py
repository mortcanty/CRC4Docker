#!/usr/bin/env python
#Name:  ex5_2.py
import sys, getopt
import numpy as np
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly
import cv2 as cv 
import matplotlib.pyplot as plt
  
def main(): 
    options,args = getopt.getopt(sys.argv[1:],'b:')
    b = 1
    for option, value in options: 
        if option == '-b':
            b = eval(value)          
    gdal.AllRegister()
    infile = args[0]      
#  read band of an MS image                  
    inDataset = gdal.Open(infile,GA_ReadOnly)     
    cols = inDataset.RasterXSize
    rows = inDataset.RasterYSize 
    rasterBand = inDataset.GetRasterBand(b) 
    band = rasterBand.ReadAsArray(0,0,cols,rows)                              
#  find and display contours    
    edges = cv.Canny(band, 20, 80)    
    _,contours,hierarchy = cv.findContours(edges,\
             cv.RETR_LIST,cv.CHAIN_APPROX_NONE)
    arr = np.zeros((rows,cols),dtype=np.uint8)
    cv.drawContours(arr, contours, -1, 255)
    plt.imshow(arr[:200,:200],cmap='gray'); plt.show()
#  determine Hu moments        
    num_contours = len(hierarchy[0])    
    hus = np.zeros((num_contours,7),dtype=np.float32)
    for i in range(num_contours): 
        arr = arr*0  
        cv.drawContours(arr, contours, i, 1)                      
        m = cv.moments(arr)
        hus[i,:] = cv.HuMoments(m).ravel()
#  plot histogram of logs of the first Hu moments    
    for i in range(3):
        idx = np.where(hus[:,i]>0) 
        hist,e=np.histogram(np.log(hus[idx,i]),50)   
        plt.plot(e[1:],hist,label='Hu moment %i'%(i+1))
    plt.legend(); plt.xlabel('log($\mu$)'); plt.show()

if __name__ == '__main__':
    main()    