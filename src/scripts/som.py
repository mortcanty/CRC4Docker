#!/usr/bin/env python
#Name:  som.py
#  Purpose:  3D Kohonen self-organizing map for multispectral image RGB visualization
#  Usage (from command line):             
#    python som.py [options] fileName
# Mort Canty (c) 2018

import numpy as np
import os, sys, getopt, time
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly, GDT_Byte

def winner(g,W):
    d = W - np.tile(g,(W.shape[0],1))
    d2 = np.sum(d**2,1)
    return np.argmin(d2)

def dsquare(c,ks,k):
    return float( (((ks-1) % c) - ((k-1) % c))**2 + (((ks-1)/c - (k-1)/c) % c)**2 + ((ks-1)/c^2 - (k-1)/c^2)**2 )

def train(G,c):
#  initialize weights with training data
    s = G.shape[0]
    idx = np.random.randint(0,s,c**3) 
    W = G[idx,:]   
#  train the network    
    etamax = 1.0
    etamin = 0.001
    sigmamax = c/2.0
    sigmamin = 0.5
    for i in range(s):
        g = G[i,:]
        kstar = winner(g,W)
        eta = etamax*(etamin/etamax)**(float(i)/s)
        sigma = sigmamax*(sigmamin/sigmamax)**(float(i)/s)
        for j in range(c**3):
            d2 = dsquare(c,kstar,j)
            lmda = np.exp(-d2/(2*sigma**2))
            W[j,:] += eta*lmda*(g-W[j,:])
    return W
            
def cluster(G,c,W):  
    m = G.shape[0]     
    som = np.zeros((m,3),dtype=np.int8) 
    for i in range(m):
        kstar = winner(G[i,:],W)
        red = kstar % c
        som[i,0] = red*255/(c-1)
        green = kstar/c % c
        som[i,1] = green*255/(c-1)
        blue = kstar/c**2
        som[i,2] = blue*255/(c-1)
    return som
            
def main():

    usage = '''
Usage: 
--------------------------------------

3D Kohonen self-organizing map for multispectral image RGB visualization

python %s [OPTIONS] filename

Options:
  -h            this help
  -p  <list>    band positions e.g. -p [1,2,3,4,5,7]
  -d  <list>    spatial subset [x,y,width,height] 
                              e.g. -d [0,0,200,200]
  -s  <int>     sample size (default 10000)
  -c  <int>     cube side length (default 5)

  -------------------------------------'''%sys.argv[0]   
            
    options,args = getopt.getopt(sys.argv[1:],'hc:s:d:p:')
    dims = None
    pos = None
    c = 5
    s = 10000
    for option, value in options:
        if option == '-h':
            print usage
            return                
        elif option == '-d':
            dims = eval(value)  
        elif option == '-p':
            pos = eval(value)  
        elif option == '-c':
            c = eval(value) 
        elif option == '-s':
            s = eval(value)         
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
    outfile = path+'/'+root+'_som'+ext      
    print '--------SOM ------------'
    print time.asctime()     
    print 'Input %s'%infile
    print 'Color cube dimension %i'%c
    start = time.time()              
    GG = np.zeros((rows*cols,len(pos))) 
    k = 0                                   
    for b in pos:
        band = inDataset.GetRasterBand(b)
        GG[:,k] = band.ReadAsArray(x0,y0,cols,rows)\
                              .astype(float).ravel()
        k += 1     
    idx = np.random.randint(0,rows*cols,s)
    G = GG[idx,:]
    print 'training...'
    W = train(G,c)
    print 'elapsed time: %s'%str(time.time()-start)
    start = time.time()
    print 'clustering...'
    som = cluster(GG,c,W)
    print 'elapsed time: %s'%str(time.time()-start)
    driver = inDataset.GetDriver() 
    outDataset = driver.Create(outfile,
                cols,rows,bands,GDT_Byte)
    projection = inDataset.GetProjection()
    geotransform = inDataset.GetGeoTransform()
    if geotransform is not None:
        gt = list(geotransform)
        gt[0] = gt[0] + x0*gt[1]
        gt[3] = gt[3] + y0*gt[5]
        outDataset.SetGeoTransform(tuple(gt))
    if projection is not None:
        outDataset.SetProjection(projection)        
    for k in range(3):        
        outBand = outDataset.GetRasterBand(k+1)
        outBand.WriteArray(np.reshape(som[:,k],(rows,cols)),0,0) 
        outBand.FlushCache() 
    print 'SOM written to: %s'%outfile      
    outDataset = None    
    inDataset = None            
    
if __name__ == '__main__':
    main()    
     
