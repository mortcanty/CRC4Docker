#!/usr/bin/env python
#Name:  kmeans.py
#  Purpose:  Perform K-means clustering
#  Usage (from command line):             
#    python kmeans.py [options] fileNmae
#
#  Copyright (c) 2018, Mort Canty

import numpy as np
import os, sys, getopt, time
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly, GDT_Byte
from scipy.cluster.vq import kmeans,vq

def main(): 
    usage = '''
Usage: 
--------------------------------------

Perform K-means clustering on multispectral imagery

python %s [OPTIONS] filename

Options:

  -h            this help
  -p  <list>    band positions e.g. -p [1,2,3,4,5,7]
  -d  <list>    spatial subset [x,y,width,height] 
                              e.g. -d [0,0,200,200]
  -k  <int>     number of clusters (default 6) 

  -------------------------------------'''%sys.argv[0]   
             
    options,args = getopt.getopt(sys.argv[1:],'hk:d:p:')
    dims = None
    pos = None
    K = 6
    for option, value in options:
        if option == '-h':
            print usage
            return                
        elif option == '-d':
            dims = eval(value)  
        elif option == '-p':
            pos = eval(value)  
        elif option == '-k':
            K = eval(value)          
    gdal.AllRegister()
    infile = args[0]     
    inDataset = gdal.Open(infile,GA_ReadOnly)     
    cols = inDataset.RasterXSize
    rows = inDataset.RasterYSize    
    bands = inDataset.RasterCount
    if dims:
        x0,y0,cols,rows = dims
    else:
        x0 = 0
        y0 = 0       
    if pos is not None:
        bands = len(pos)
    else:
        pos = range(1,bands+1)        
    path = os.path.dirname(infile)
    basename = os.path.basename(infile)
    root, ext = os.path.splitext(basename)
    outfile = path+'/'+root+'_kmeans'+ext      
    print '------------ k-means ------------'
    print time.asctime()     
    print 'Input %s'%infile
    print 'Number of clusters %i'%K
    start = time.time()               
    G = np.zeros((rows*cols,bands)) 
    k = 0                                   
    for b in pos:
        band = inDataset.GetRasterBand(b)
        G[:,k] = band.ReadAsArray(x0,y0,cols,rows)\
                              .astype(float).ravel()
        k += 1        
    centers, _ = kmeans(G,K)
    labels, _ = vq(G,centers)      
    driver = gdal.GetDriverByName('GTiff')   
    outDataset = driver.Create(outfile,
                    cols,rows,1,GDT_Byte)         
    outBand = outDataset.GetRasterBand(1)
    outBand.WriteArray(np.reshape(labels+1,
                            (rows,cols)),0,0) 
    outBand.FlushCache() 
    outDataset = None    
    inDataset = None   
    print 'Kmeans result written to: %s'%outfile 
    print 'elapsed time: %s'%str(time.time()-start)        
 
if __name__ == '__main__':
    main()    