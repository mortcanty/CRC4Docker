#!/usr/bin/env python
#******************************************************************************
#  Name:     krx.py
#  Purpose:  Kernel RX anomaly detection for multi- and hyperspectral images
#  Usage:             
#    python krx.py [options] filename
#
#  Copyright (c) 2018, Mort Canty

import numpy as np
import auxil.auxil1 as auxil
import os, sys, getopt, time
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly,GDT_Float32
 
def main():      
    usage = '''
Usage:
------------------------------------------------

Kernel RX anomaly detection for multi- and hyperspectral images

python %s [OPTIONS]  filename

Options:
  
  -h         this help
   
  -s  <int>   sample size for kernel matrix (default 1000)
  
  -n  <int>   nscale parameter for Gauss kernel (default 10)

-------------------------------------------------'''%sys.argv[0]      
    
    
    
    
    options,args = getopt.getopt(sys.argv[1:],'hs:n:')
    m = 1000
    nscale = 10
    for option, value in options: 
        if option == '-h':
            print usage
            return    
        elif option == '-s':
            m = eval(value)
        elif option == '-n':
            nscale = eval(value)     
    gdal.AllRegister()
    infile = args[0] 
    path = os.path.dirname(infile)
    basename = os.path.basename(infile)
    root, ext = os.path.splitext(basename)
    outfile = path+'/'+root+'_krx'+ext        
#  input image                   
    inDataset = gdal.Open(infile,GA_ReadOnly)
    cols = inDataset.RasterXSize
    rows = inDataset.RasterYSize 
    bands = inDataset.RasterCount
    projection = inDataset.GetProjection()
    geotransform = inDataset.GetGeoTransform()  
#  image data matrix    
    GG = np.zeros((rows*cols,bands))                               
    for b in range(bands):
        band = inDataset.GetRasterBand(b+1)
        GG[:,b] = band.ReadAsArray(0,0,cols,rows)\
                              .astype(float).ravel()          
    inDataset = None
#  random training data matrix
    idx = np.random.randint(0,rows*cols,size=m)
    G = GG[idx,:]
#  KRX-algorithm        
    print '------------ KRX ---------------'
    print time.asctime()     
    print 'Input %s'%infile
    start = time.time()  
    K,gma=auxil.kernelMatrix(G,nscale=nscale,kernel=1)
    Kc = auxil.center(K)
    print 'GMA: %f'%gma
#  pseudoinvert centered kernel matrix     
    lam, alpha = np.linalg.eigh(Kc)
    idx = range(m)[::-1]
    lam = lam[idx]
    alpha = alpha[:,idx]
    tol = max(lam)*m*np.finfo(float).eps
    r = np.where(lam>tol)[0].shape[0]
    alpha = alpha[:,:r]
    lam = lam[:r]
    Kci = alpha*np.diag(1./lam)*alpha.T
#  row-by-row anomaly image    
    res = np.zeros((rows,cols))
    Ku = np.sum(K,0)/m - np.sum(K)/m**2
    Ku = np.mat(np.ones(cols)).T*Ku
    for i in range(rows):
        if i % 100 == 0:
            print 'row: %i'%i     
        GGi = GG[i*cols:(i+1)*cols,:]
        Kg,_=auxil.kernelMatrix(GGi,G,gma=gma,kernel=1)
        a = np.sum(Kg,1)
        a = a*np.mat(np.ones(m))
        Kg = Kg - a/m
        Kgu = Kg - Ku
        d = np.sum(np.multiply(Kgu,Kgu*Kci),1) 
        res[i,:] = d.ravel()  
#  output 
    driver = gdal.GetDriverByName('GTiff')    
    outDataset = driver.Create(outfile,cols,rows,1,\
                                    GDT_Float32) 
    if geotransform is not None:
        outDataset.SetGeoTransform(geotransform)
    if projection is not None:
        outDataset.SetProjection(projection) 
    outBand = outDataset.GetRasterBand(1) 
    outBand.WriteArray(np.asarray(res,np.float32),0,0) 
    outBand.FlushCache()
    outDataset = None   
    print 'Result written to %s'%outfile
    print 'elapsed time: %s'%str(time.time()-start)
    
if __name__ == '__main__':
    main()