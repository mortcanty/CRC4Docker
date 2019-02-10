#!/usr/bin/env python
#Name:  ekmeans.py
#  Purpose:  Perform extended K-means clustering
#  Usage (from command line):             
#    python ekmeans.py [options] fileNmae
#
#  Copyright (c) 2018, Mort Canty

import numpy as np
import os, sys, getopt, time
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly, GDT_Byte

def main(): 
    
    usage = '''            
Usage: 
--------------------------------------

Perform extended K-means clustering

python %s [OPTIONS] filename

Options:
  -h            this help
  -b  <int>     band position (default 1)
  -d  <list>    spatial subset [x,y,width,height] 
                              e.g. -d [0,0,200,200]
  -k  <int>     number of metaclusters (default 8)
  
  -------------------------------------'''%sys.argv[0]       
               
    options,args = getopt.getopt(sys.argv[1:],'hk:d:b:p:')
    dims = None
    b = 1
    K = 8
    for option, value in options:
        if option == '-h':
            print usage
            return                
        elif option == '-d':
            dims = eval(value)  
        elif option == '-b':
            b = eval(value)  
        elif option == '-k':
            K = eval(value)          
    gdal.AllRegister()
    infile = args[0]     
    inDataset = gdal.Open(infile,GA_ReadOnly)     
    cols = inDataset.RasterXSize
    rows = inDataset.RasterYSize    
    if dims:
        x0,y0,cols,rows = dims
    else:
        x0 = 0
        y0 = 0        
    path = os.path.dirname(infile)
    basename = os.path.basename(infile)
    root, ext = os.path.splitext(basename)
    outfile = path+'/'+root+'_ekmeans'+ext      
    print '------- extended k-means ---------'
    print time.asctime()     
    print 'Input: %s'%infile
    print 'Band: %i'%b
    print 'Meta-clusters: %i'%K
    start = time.time()        
    m = rows*cols       
    band = inDataset.GetRasterBand(b)
    G =band.ReadAsArray(x0,y0,cols,rows)
    labels = np.zeros(m)
    sigma2 = np.std(G - np.roll(G,(0,1)))**2
    alphaE = -1/(2*np.log(1.0/K))
    hist, _ = np.histogram(G,bins = 256) 
    indices = np.where(hist>0)[0]
    K = indices.size
    means = np.array(range(256))[indices] 
    priors = hist[indices]/np.float(m)   
    delta = 100.0
    itr = 0
    G = G.ravel()
    while (delta>1.0) and (itr<100):
        print 'Clusters: %i delta: %f'%(K,delta)
        indices = np.where(priors>0.01)[0]
        K = indices.size 
        ds = np.zeros((K,m))
        means = means[indices]
        priors = priors[indices]
        means1 = means
        priors1 = priors
        means = means*0.0
        priors = priors*0.0
        for j in range(K):
            ds[j,:] = (G-means1[j])**2/(2*sigma2) \
                      - alphaE*np.log(priors1[j]) 
        min_ds = np.min(ds,axis=0)
        for j in range(K):
            indices = np.where(ds[j,:] == min_ds)[0]
            if indices.size>0:
                mj = indices.size
                priors[j] = mj/np.float(m)
                means[j] = np.sum(G[indices])/mj
                labels[indices] = j
        delta = np.max(np.abs(means-means1))
        itr += 1       
    driver = gdal.GetDriverByName('GTiff')   
    outDataset = driver.Create(outfile,
                    cols,rows,1,GDT_Byte)         
    outBand = outDataset.GetRasterBand(1)
    outBand.WriteArray(np.reshape(labels+1,(rows,cols)),0,0) 
    outBand.FlushCache() 
    outDataset = None    
    inDataset = None   
    print 'Extended K-means result written to: %s'%outfile 
    print 'elapsed time: %s'%str(time.time()-start)        
 
if __name__ == '__main__':
    main()    