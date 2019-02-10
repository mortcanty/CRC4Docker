#!/usr/bin/env python
#******************************************************************************
#  Name:     scatterplot.py
#  Purpose:  
#    Display a scatterplot of two ms bands      via similarity warping.
#
#  Usage:        
#    python scatterplot.py [OPTIONS] filename1 [filename2] band1 band2
#
#  Copyright (c) 2018 Mort Canty

import numpy as np
import sys, getopt
from osgeo import gdal
import matplotlib.pyplot as plt
from osgeo.gdalconst import GA_ReadOnly
  
    
def main(): 
    usage = '''
    Usage:
------------------------------------------------

Display a scatterplot   

python %s [OPTIONS] filename1 [filename2] band1 band2
      
Options:

   -h          this help
   -d <list>   spatial subset
   -n <int>    samples (default 10000)
   -s <string> save in eps format ''' %sys.argv[0]
   
    options, args = getopt.getopt(sys.argv[1:],'hd:s:')  
    dims = None
    samples = 10000
    sfn = None
    for option, value in options:
        if option == '-h':
            print usage
            return        
        elif option == '-d':
            dims = eval(value) 
        elif option == '-n':
            samples = eval(value) 
        elif option == '-s':
            sfn = value            
    if len(args)==4:     
        fn1 = args[0] 
        fn2 = args[1]  
        b1 = eval(args[2])
        b2 = eval(args[3])
    elif len(args)==3:
        fn1 = args[0] 
        fn2 = args[0]  
        b1 = eval(args[1])
        b2 = eval(args[2])
    else:
        print 'Incorrect number of arguments'
        print usage
        sys.exit(1)             
    gdal.AllRegister()
    inDataset = gdal.Open(fn1,GA_ReadOnly)  
    cols = inDataset.RasterXSize
    rows = inDataset.RasterYSize     
    if dims:
        x0,y0,cols,rows = dims
    else:
        x0 = 0
        y0 = 0          
    g1 = inDataset.GetRasterBand(b1).ReadAsArray(x0,y0,cols,rows)\
                              .astype(float).ravel()
    inDataset = gdal.Open(fn2,GA_ReadOnly)  
    cols = inDataset.RasterXSize
    rows = inDataset.RasterYSize     
    if dims:
        x0,y0,cols,rows = dims
    else:
        x0 = 0
        y0 = 0          
    g2 = inDataset.GetRasterBand(b2).ReadAsArray(x0,y0,cols,rows)\
                              .astype(float).ravel()   
    idx = np.random.randint(0,rows*cols,samples)
    plt.plot(g1[idx],g2[idx],'.')
    if sfn is not None:
        plt.savefig(sfn,bbox_inches='tight')  
    plt.show()
   
if __name__ == '__main__':
    main()    