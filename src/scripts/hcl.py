#!/usr/bin/env python
#Name:  hcl.py
#  Purpose:  Perform agglomerative hierarchical clustering
#  Usage (from command line):             
#    python hcl.py [options] fileNmae
#
#  Copyright (c) 2018, Mort Canty

import numpy as np
import os, sys, getopt, time
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly, GDT_Byte
import matplotlib.pyplot as plt
import auxil.supervisedclass as sc

def best_merge(D):
#  return the best merge, larger index first    
    n = D.shape[0]
    label = np.argmin(D)
    return [np.mod(label,n),label/n]

def main(): 

    usage = '''            
Usage: 
--------------------------------------

Perform agglomerative hierarchical clustering

python %s [OPTIONS] filename

Options:
  -h            this help
  -p  <list>    band positions e.g. -p [1,2,3,4,5,7]
  -d  <list>    spatial subset [x,y,width,height] 
                              e.g. -d [0,0,200,200]
  -k  <int>     number of clusters (default 8)
  -s  <int>     number of samples (default 1000)

  -------------------------------------'''%sys.argv[0]
            
            
    options,args = getopt.getopt(sys.argv[1:],'hs:k:d:p:')
    dims = None
    pos = None
    K = 8
    m = 1000
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
        elif option == '-s':
            m = eval(value)              
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
    outfile = path+'/'+root+'_hcl'+ext      
    print '------- Hierarchical clustering ---------'
    print time.asctime()     
    print 'Input: %s'%infile
    print 'Clusters: %i'%K
    print 'Samples: %i'%m
    start = time.time()  
    GG = np.zeros((cols*rows,bands))      
    k = 0
    for b in pos:
        band = inDataset.GetRasterBand(b)
        band = band.ReadAsArray(x0,y0,cols,rows).astype(float)
        GG[:,k] = np.ravel(band)
        k += 1   
#  training data
    idx = np.random.randint(0,rows*cols,size=m)
    G = GG[idx,:] 
    ones = np.mat(np.ones(m,dtype=np.int))
    mults = np.ones(m,dtype=np.int)
    Ls = np.array(range(m))  
#  initial cost array    
    G2 = np.mat(np.sum(G**2,1))
    Delta = G2.T*ones
    Delta = Delta + Delta.T 
    Delta = Delta - 2*np.mat(G)*np.mat(G).T
    idx = np.tril_indices(m)
    Delta[idx] = 10e10
    Delta = Delta.A
#  begin iteration    
    cost = 0.0
    costarray = []
    c = m
    while c > K:
        bm = best_merge(Delta)
        j = bm[0]
        i = bm[1]
#      j > i        
        costarray.append(cost + Delta[i,j])
#      re-label   
        idx = np.where(Ls == j)[0]    
        Ls[idx] = i
        idx = np.where(Ls > j)[0]
        Ls[idx] -= 1
#      pre-merge multiplicities
        ni = mults[i]
        nj = mults[j]        
#      update merge-cost array, k = i+1 ... c-1        
        if c-i-1 == 0:
            k = [i+1] 
        else:
            k = i+1+range(c-i-1)
        nk = mults[k]
        Dkj = np.minimum(Delta[k,j].ravel(),Delta[j,k].ravel())       
        idx = np.where(k == j)[0] 
        Dkj[idx]=0
        Delta[i,k] = ( (ni+nk)*Delta[i,k]+(nj+nk)*Dkj-nk*Delta[i,j] )/(ni+nj+nk)
#     update merge-cost array, k = 0 ... i-1 
        if i == 0:
            k = [0]
        else:
            k = range(i-1)
        nk = mults[k]
        Dkj = np.minimum(Delta[k,j].ravel(),Delta[j,k].ravel()) 
        idx = np.where(k == j)[0] 
        Dkj[idx]=0    
        Delta[k,i] = ( (ni+nk)*Delta[k,i]+(nj+nk)*Dkj-nk*Delta[i,j] )/(ni+nj+nk)    
#      update multiplicities
        mults[i] = mults[i]+mults[j]
#      delete the upper cluster
        idx =np.ones(c)       
        idx[j] = 0
        idx = np.where(idx == 1)[0]
        mults = mults[idx]
        Delta = Delta[:,idx]
        Delta = Delta[idx,:]
        c -= 1
    print 'classifying...'
    labs = []
    for L in Ls:
        lab = np.zeros(K)
        lab[L] = 1.0
        labs.append(lab) 
    labs = np.array(labs)       
    classifier = sc.Maxlike(G,labs) 
    if  classifier.train():     
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
        cls, _ = classifier.classify(GG)
        outBand = outDataset.GetRasterBand(1)
        outBand.WriteArray(np.reshape(cls,(rows,cols)),0,0) 
        outBand.FlushCache() 
        outDataset = None
        inDataset = None       
        ymax = np.max(costarray)    
        plt.loglog(range(K,m),list(reversed(costarray)))
        p = plt.gca()
        p.set_title('Merge cost')
        p.set_xlabel('Custers') 
        p.set_ylim((1,ymax))    
        plt.show()    
        print 'result written to: '+outfile    
        print 'elapsed time: '+str(time.time()-start)
    else:
        print 'classification failed'                  
       
if __name__ == '__main__':
    main()    