#!/usr/bin/env python
#******************************************************************************
#  Name:     kPCA.py
#  Purpose:  Perform kernel PCA on multispectral imagery 
#  Usage:             
#    python kpca.py 

import auxil.auxil1 as auxil
import sys, os, time, getopt
import numpy as np
from scipy import linalg
from scipy.cluster.vq import kmeans
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly, GDT_Float32
import matplotlib.pyplot as plt
    
def main():
    usage = '''Usage: python %s [-h] [-n] [-d dims] [-p pos] [-s sample size] \n
           [-e number of eigenvectors (kPCs, default 10)] -[k kernel] fileName\n
            spatial and spectral dimensions are lists, e.g., -d [0,0,400,400] \n
            use -k 0 for linear, -k 1 for Gaussian kernel (default) \n
            use -s 0 to use kmeans sampling with 100 clusters (default) \n
            use -n to disable graphics output'''%sys.argv[0]
    options,args = getopt.getopt(sys.argv[1:],'hnd:p:s:e:k:')
    dims = None
    pos = None
    graphics = True
    m = 0
    n = 10
    kernel = 1
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
        elif option == '-s':
            m = eval(value)   
        elif option == '-e':
            n = eval(value) 
        elif option == '-k':
            kernel = eval(value)    
    gdal.AllRegister()
    infile = args[0] 
    path = os.path.dirname(infile)
    basename = os.path.basename(infile)
    root, ext = os.path.splitext(basename)
    outfile = path+'/'+root+'_kpca'+ext      
    if infile:                   
        inDataset = gdal.Open(infile,GA_ReadOnly)     
        cols = inDataset.RasterXSize
        rows = inDataset.RasterYSize    
        bands = inDataset.RasterCount
    else:
        return
    if dims:
        x0,y0,cols,rows = dims
    else:
        x0 = 0
        y0 = 0 
    if pos is not None:
        bands = len(pos)
    else:
        pos = range(1,bands+1)      
    print '========================='
    print '       kPCA'
    print '========================='
    print 'infile:  '+infile
    print 'samples: '+str(m) 
    if kernel == 0:
        print 'kernel:  '+'linear' 
    else:
        print 'kernel:  '+'Gaussian'  
    start = time.time()                                     
    if kernel == 0:
        n = min(bands,n)
# construct data design matrices           
    XX = np.zeros((cols*rows,bands))      
    k = 0
    for b in pos:
        band = inDataset.GetRasterBand(b)
        band = band.ReadAsArray(x0,y0,cols,rows).astype(float)
        XX[:,k] = np.ravel(band)
        k += 1   
    if m > 0:   
        idx = np.fix(np.random.random(m)*(cols*rows)).astype(np.integer)
        X = XX[idx,:]  
    else:   
        print 'running k-means on 100 cluster centers...'
        X,_ = kmeans(XX,100,iter=1)
        m = 100
    print 'centered kernel matrix...'
# centered kernel matrix    
    K, gma = auxil.kernelMatrix(X,kernel=kernel)      
    meanK = np.sum(K)/(m*m)
    rowmeans = np.mat(np.sum(K,axis=0)/m)
    if gma is not None:
        print 'gamma: '+str(round(gma,6))    
    K = auxil.center(K)    
    print 'diagonalizing...'
# diagonalize
    try:
        w, v = linalg.eigh(K,eigvals=(m-n,m-1)) 
        idx = np.argsort(w)[::-1]
        w = w[idx]
        v = v[:,idx] 
#      variance of kPCs        
        var = w/m
    except linalg.LinAlgError:
        print 'eigenvalue computation failed'
        sys.exit()   
#  dual variables (normalized eigenvectors)
    alpha = np.mat(v)*np.mat(np.diag(1/np.sqrt(w)))           
    print 'projecting...' 
#  projecting     
    image = np.zeros((rows,cols,n)) 
    for i in range(rows):
        XXi = XX[i*cols:(i+1)*cols,:]       
        KK,gma = auxil.kernelMatrix(X,XXi,kernel=kernel,gma=gma)
#  centering on training data: 
#      subtract column means
        colmeans = np.mat(np.sum(KK,axis=0)/m)
        onesm = np.mat(np.ones(m))
        KK = KK - onesm.T*colmeans
#      subtract row means
        onesc = np.mat(np.ones(cols))                
        KK = KK - rowmeans.T*onesc
#      add overall mean
        KK = KK + meanK 
#      project
        image[i,:,:] = KK.T*alpha    
#  write to disk
    driver = inDataset.GetDriver()    
    outDataset = driver.Create(outfile,cols,rows,n,GDT_Float32)
    projection = inDataset.GetProjection()
    geotransform = inDataset.GetGeoTransform()
    if geotransform is not None:
        gt = list(geotransform)
        gt[0] = gt[0] + x0*gt[1]
        gt[3] = gt[3] + y0*gt[5]
        outDataset.SetGeoTransform(tuple(gt))
    if projection is not None:
        outDataset.SetProjection(projection)        
    for k in range(n):        
        outBand = outDataset.GetRasterBand(k+1)
        outBand.WriteArray(image[:,:,k],0,0) 
        outBand.FlushCache() 
    outDataset = None
    inDataset = None
    print 'result written to: '+outfile    
    print 'elapsed time: '+str(time.time()-start)
    if graphics:                        
        plt.plot(range(1,n+1), var,'b-')
        plt.title('Kernel PCA') 
        plt.xlabel('Principal Component')
        plt.ylabel('Variance')        
        plt.show()        
    print '--done------------------------'  
       
if __name__ == '__main__':
    main()    