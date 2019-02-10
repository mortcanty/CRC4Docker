#!/usr/bin/env python
#******************************************************************************
#  Name:     mnf.py
#  Purpose:  Minimum noise fraction
#  Usage (from command line):             
#    python mnf.py  [options] fileNmae
#
#  Copyright (c) 2018 Mort Canty

import numpy as np
import auxil.auxil1 as auxil
import os, sys, getopt, time
from osgeo import gdal
import matplotlib.pyplot as plt
from osgeo.gdalconst import GA_ReadOnly,GDT_Float32

def main():            
    usage = '''            
Usage: 
--------------------------------------

Minimum noise fraction

python %s [OPTIONS] filename

Options:
  -h            this help
  -p  <list>    band positions e.g. -p [1,2,3,4,5,7]
  -d  <list>    spatial subset [x,y,width,height] 
                              e.g. -d [0,0,200,200]
  -r  <int>     number of components for reconstruction 
  -n            disable graphics   
  
  -------------------------------------'''%sys.argv[0]           
              
    options,args = getopt.getopt(sys.argv[1:],'hnd:p:')
    dims = None
    pos = None
    graphics = True
    for option, value in options: 
        if option == '-h':
            print usage
            return      
        elif option == '-n':
            graphics = False
        elif option == '-d':
            dims = eval(value)  
        elif option == '-p':
            pos = eval(value)
    gdal.AllRegister()   
    infile = args[0] 
    path = os.path.dirname(infile)
    basename = os.path.basename(infile)
    root, ext = os.path.splitext(basename)
    outfile = path+'/'+root+'_mnf'+ext  
    print '------------MNF ---------------'
    print time.asctime()     
    print 'Input %s'%infile
    start = time.time()    
    
    inDataset = gdal.Open(infile,GA_ReadOnly)
    try:                         
        cols = inDataset.RasterXSize
        rows = inDataset.RasterYSize    
        bands = inDataset.RasterCount
    except Exception as e:
        print 'Error: %s  -- Could not read in file'%e
        sys.exit(1)
    if dims:
        x0,y0,cols,rows = dims
    else:
        x0 = 0
        y0 = 0       
    if pos is not None:
        bands = len(pos)
    else:
        pos = range(1,bands+1)
#  data matrix
    G = np.zeros((rows*cols,bands))
    k = 0                               
    for b in pos:
        band = inDataset.GetRasterBand(b)
        tmp = band.ReadAsArray(x0,y0,cols,rows)\
                              .astype(float).ravel()
        G[:,k] = tmp - np.mean(tmp)
        k += 1      
#  covariance matrix
    S = np.mat(G).T*np.mat(G)/(cols*rows-1)   
#  noise covariance matrix    
    D = np.zeros((cols*rows,bands))
    tmp = np.reshape(G,(rows,cols,bands))    
    for b in range(bands):
        tmp = np.reshape(G[:,b],(cols,rows))
        D[:,b] = (tmp-(np.roll(tmp,1,axis=0)+np.roll(tmp,1,axis=1))/2).ravel() 
    Sn = np.mat(D).T*np.mat(D)/(2*(rows*cols-1))
#  generalized eigenproblem    
    lambdas,eivs = auxil.geneiv(Sn,S)
    idx = (np.argsort(lambdas))
    lambdas = lambdas[idx]
    eivs = (eivs[:,idx]).T      
    mnfs = np.array(G*eivs)
    mnfs = np.reshape(mnfs,(rows,cols,bands)) 
#  write to disk       
    driver = inDataset.GetDriver() 
    outDataset = driver.Create(outfile,
                cols,rows,bands,GDT_Float32)
    projection = inDataset.GetProjection()
    geotransform = inDataset.GetGeoTransform()
    if geotransform is not None:
        gt = list(geotransform)
        gt[0] = gt[0] + x0*gt[1]
        gt[3] = gt[3] + y0*gt[5]
        outDataset.SetGeoTransform(tuple(gt))
    if projection is not None:
        outDataset.SetProjection(projection)        
    for k in range(bands):        
        outBand = outDataset.GetRasterBand(k+1)
        outBand.WriteArray(mnfs[:,:,k],0,0) 
        outBand.FlushCache() 
    snrs = 1.0/lambdas-1.0
    print 'Signal to noise ratios: %s'%str(snrs)
    if graphics: 
        plt.plot(range(1,bands+1),snrs)
        plt.title(infile)
        plt.ylabel('Signal to noise') 
        plt.xlabel('Spectral Band')
        plt.show()
        plt.close()         
    print 'MNFs written to: %s'%outfile      
    outDataset = None    
    inDataset = None           
    print 'elapsed time: %s'%str(time.time()-start) 
     
if __name__ == '__main__':
    main()    