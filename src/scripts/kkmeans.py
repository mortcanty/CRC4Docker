#!/usr/bin/env python
#******************************************************************************
#  Name:     kkmeans.py
#  Purpose:  Perform kernel K-means clustering on multispectral imagery 
#  Usage:             
#    python kkmeans.py 
#
#  Copyright (c) 2018, Mort Canty

import numpy as np
import os, sys, getopt, time
from osgeo import gdal
import auxil.auxil1 as auxil
from osgeo.gdalconst import GA_ReadOnly, GDT_Byte

def main():
 
    usage = '''            
Usage: 
--------------------------------------

Perform kernel K-means clustering on multispectral imagery

python %s [OPTIONS] filename

Options:
  -h            this help
  -p  <list>    band positions e.g. -p [1,2,3,4,5,7]
  -d  <list>    spatial subset [x,y,width,height] 
                              e.g. -d [0,0,200,200]
  -k  <int>     number of clusters (default 6)
  -m  <int>     number of samples (default 1000)
  -n  <int>     nscale for Gauss kernel (default 1)

  -------------------------------------'''%sys.argv[0]   
  
    options,args = getopt.getopt(sys.argv[1:],'hk:n:m:d:p:')
    dims = None
    pos = None
    K = 6
    m = 1000
    nscale = 1
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
        elif option == '-m':
            m = eval(value)   
        elif option == '-n':
            nscale = eval(value)         
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
    outfile = path+'/'+root+'_kkmeans'+ext        
    print '========================='
    print '    kernel k-means'
    print '========================='
    print 'infile: %s'%infile
    print 'samples: %i '%m 
    print 'clusters: %i'%K  
    start = time.time()                                     
#  input data matrix           
    XX = np.zeros((cols*rows,bands))      
    k = 0
    for b in pos:
        band = inDataset.GetRasterBand(b)
        band = band.ReadAsArray(x0,y0,cols,rows).astype(float)
        XX[:,k] = np.ravel(band)
        k += 1
#  training data matrix
    idx = np.random.randint(0,rows*cols,size=m)
    X = XX[idx,:]  
    print 'kernel matrix...'
# uncentered kernel matrix    
    KK, gma = auxil.kernelMatrix(X,kernel=1,nscale=nscale)      
    if gma is not None:
        print 'gamma: '+str(round(gma,6))    
#  initial (random) class labels
    labels = np.random.randint(K,size = m)  
#  iteration
    change = True
    itr = 0
    onesm = np.mat(np.ones(m,dtype=float))
    while change and (itr < 100):
        change = False
        U = np.zeros((K,m))
        for i in range(m):
            U[labels[i],i] = 1
        M =  np.diag(1.0/(np.sum(U,axis=1)+1.0))
        MU = np.mat(np.dot(M,U))
        Z = (onesm.T)*np.diag(MU*KK*(MU.T))-2*KK*(MU.T)
        Z = np.array(Z) 
        labels1 = (np.argmin(Z,axis=1) % K).ravel()
        if np.sum(labels1 != labels):
            change = True
        labels = labels1   
        itr += 1
    print 'iterations: %i'%itr 
#  classify image
    print 'classifying...'
    i = 0
    A = np.diag(MU*KK*(MU.T))
    A = np.tile(A,(cols,1))
    class_image = np.zeros((rows,cols),dtype=np.byte)
    while i < rows:     
        XXi = XX[i*cols:(i+1)*cols,:]
        KKK,_ = auxil.kernelMatrix(X,XXi,gma=gma,kernel=1)
        Z = A - 2*(KKK.T)*(MU.T)
        Z= np.array(Z)
        labels = np.argmin(Z,axis=1).ravel()
        class_image[i,:] = (labels % K) +1
        i += 1   
    sys.stdout.write("\n")    
#  write to disk
    driver = gdal.GetDriverByName('GTiff')    
    outDataset = driver.Create(outfile,cols,rows,1,GDT_Byte)
    projection = inDataset.GetProjection()
    geotransform = inDataset.GetGeoTransform()
    if geotransform is not None:
        gt = list(geotransform)
        gt[0] = gt[0] + x0*gt[1]
        gt[3] = gt[3] + y0*gt[5]
        outDataset.SetGeoTransform(tuple(gt))
    if projection is not None:
        outDataset.SetProjection(projection)               
    outBand = outDataset.GetRasterBand(1)
    outBand.WriteArray(class_image,0,0) 
    outBand.FlushCache() 
    outDataset = None
    inDataset = None
    print 'result written to: '+outfile    
    print 'elapsed time: '+str(time.time()-start)                        
       
if __name__ == '__main__':
    main()    