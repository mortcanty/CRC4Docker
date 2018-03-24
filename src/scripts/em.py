#!/usr/bin/env python
#******************************************************************************
#  Name:     em.py
#  Purpose:  Perform Gaussian mixture clustering on multispectral imagery 
#  Usage:             
#    python em.py 

import auxil.auxil as auxil
import auxil.header as header 
from auxil.auxil import ctable
import os, sys, time, getopt
import numpy as np
import scipy.ndimage.interpolation as ndi
import scipy.ndimage.filters as ndf
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly, GDT_Byte

def em(G,U,T0,beta,rows,cols,unfrozen=None):
    K,n = U.shape
    N = G.shape[1]
    if unfrozen is None:
        unfrozen = range(n)
    V = U*0.0
    Nb = np.array([[0.0,0.25,0.0],[0.25,0.0,0.25],[0.0,0.25,0.0]])
    Cs = np.zeros((K,N,N))
    pdens = np.zeros(K)
    fhv = np.zeros(K)
    dU = 1.0
    itr = 0
    T = T0
    print 'running EM on %i pixel vectors'%n
    while ((dU > 0.001) or (itr < 10)) and (itr < 500):
        Uold = U+0.0
        ns = np.sum(U,axis=1)
#      prior probabilities
        Ps = np.asarray(ns/n).ravel()
#      cluster means
        Ms = np.asarray((np.mat(U)*np.mat(G)).T)
#      loop over the cluster index
        for k in range(K):
            Ms[:,k] = Ms[:,k]/ns[k]
            W = np.tile(Ms[:,k].ravel(),(n,1))
            Ds = G - W
#          covariance matrix
            for i in range(N):
                W[:,i] = np.sqrt(U[k,:]) \
                    .ravel()*Ds[:,i].ravel()             
            C = np.mat(W).T*np.mat(W)/ns[k]
            Cs[k,:,:] = C
            sqrtdetC = np.sqrt(np.linalg.det(C))
            Cinv = np.linalg.inv(C)
            qf = np.asarray(np.sum(np.multiply(Ds \
                        ,np.mat(Ds)*Cinv),1)).ravel()
#          class hypervolume and partition density
            fhv[k] = sqrtdetC
            idx = np.where(qf < 1.0)
            pdens[k] = np.sum(U[k,idx])/fhv[k]
#          new memberships
            U[k,unfrozen] = np.exp(-qf[unfrozen]/2.0)\
                   *(Ps[k]/sqrtdetC)
#          random membership for annealing
            if T > 0.0:
                Ur = 1.0 - np.random\
                     .random(len(unfrozen))**(1.0/T)
                U[k,unfrozen] = U[k,unfrozen]*Ur               
#      spatial membership            
        if beta > 0:            
#          normalize the class probabilities prior to convolving         
            a = np.sum(U,axis=0)
            idx = np.where(a == 0)[0]
            a[idx] = 1.0
            for k in range(K):
                U[k,:] = U[k,:]/a                        
            for k in range(K):
                U_N = 1.0 - ndf.convolve(np.reshape(U[k,:],(rows,cols)),Nb)
                V[k,:] = np.exp(-beta*U_N).ravel()                        
#          combine spectral/spatial
            U[:,unfrozen] = U[:,unfrozen]*V[:,unfrozen] 
#      normalize all
        a = np.sum(U,axis=0)
        idx = np.where(a == 0)[0]
        a[idx] = 1.0
        for k in range(K):
            U[k,:] = U[k,:]/a                
        T = 0.8*T 
#      log likelihood
        Uflat = U.ravel()
        Uoldflat = Uold.ravel()
        idx = np.where(U.flat)[0]
        loglike = np.sum(Uoldflat[idx]*np.log(Uflat[idx]))
        dU = np.max(Uflat-Uoldflat)  
        if (itr % 10) == 0:
            print 'em iteration %i: dU: %f loglike: %f'%(itr,dU,loglike)
        itr += 1         
    return (U,np.transpose(Ms),Cs,Ps,pdens)
                                                  
                                        
def main():
    usage = '''
Usage: 
---------------------------------------------------------
python %s  [-p "bandPositions"] [-d "spatialDimensions"] 
[-K number of clusters] [-M max scale][-m min scale] 
[-t initial annealing temperature] [-s spatial mixing factor] 
[-P generate class probabilities image] filename

bandPositions and spatialDimensions are lists, 
e.g., -p [1,2,4] -d [0,0,400,400]  

If the input file is named 

         path/filenbasename.ext then

The output classification file is named 

         path/filebasename_em.ext

and the class probabilities output file is named

         path/filebasename_emprobs.ext
--------------------------------------------------------''' %sys.argv[0]
    options, args = getopt.getopt(sys.argv[1:],'hp:d:K:M:m:t:s:P')
    pos = None
    dims = None  
    K,max_scale,min_scale,T0,beta,probs = (None,None,None,None,None,None)        
    for option, value in options:
        if option == '-h':
            print usage
            return
        elif option == '-p':
            pos = eval(value)
        elif option == '-d':
            dims = eval(value) 
        elif option == '-K':
            K = eval(value)
        elif option == '-M':
            max_scale = eval(value)
        elif option == '-m':
            min_scale = eval(value)  
        elif option == '-t':
            T0 = eval(value)
        elif option == '-s':
            beta = eval(value) 
        elif option == '-P':
            probs = True                              
    if len(args) != 1: 
        print 'Incorrect number of arguments'
        print usage
        sys.exit(1)       
    if K is None:
        K = 6
    if max_scale is None:
        max_scale = 2   
    else:
        max_scale = min((max_scale,3))  
    if min_scale is None:
        min_scale = 0   
    else:
        min_scale = min((max_scale,min_scale)) 
    if T0 is None:
        T0 = 0.5   
    if beta is None:
        beta = 0.5   
    if probs is None:
        probs = False
                                                  
    gdal.AllRegister()
    infile = args[0]
    
    gdal.AllRegister() 
    try:                   
        inDataset = gdal.Open(infile,GA_ReadOnly)     
        cols = inDataset.RasterXSize
        rows = inDataset.RasterYSize    
        bands = inDataset.RasterCount
    except Exception as e:
        print 'Error: %s  --Image could not be read'%e
        sys.exit(1)
    if pos is not None:
        bands = len(pos)
    else:
        pos = range(1,bands+1)
    if dims:
        x0,y0,cols,rows = dims
    else:
        x0 = 0
        y0 = 0   
    class_image = np.zeros((rows,cols),dtype=np.byte)   
    path = os.path.dirname(infile)
    basename = os.path.basename(infile)
    root, ext = os.path.splitext(basename)
    outfile = path+'/'+root+'_em'+ext
    if probs:
        probfile = path+'/'+root+'_emprobs'+ext
    print '--------------------------'
    print '     EM clustering'
    print '--------------------------'
    print 'infile:   %s'%infile
    print 'clusters: %i'%K
    print 'T0:       %f'%T0
    print 'beta:     %f'%beta         

    start = time.time()                                     
#  read in image and compress 
    path = os.path.dirname(infile) 
    basename = os.path.basename(infile)
    root, ext = os.path.splitext(basename)
    DWTbands = []               
    for b in pos:
        band = inDataset.GetRasterBand(b)
        DWTband = auxil.DWTArray(band.ReadAsArray(x0,y0,cols,rows).astype(float),cols,rows)
        for i in range(max_scale):
            DWTband.filter()
        DWTbands.append(DWTband)
    rows,cols = DWTbands[0].get_quadrant(0).shape    
    G = np.transpose(np.array([DWTbands[i].get_quadrant(0,float=True).ravel() for i in range(bands)]))
#  initialize membership matrix    
    n = G.shape[0]
    U = np.random.random((K,n))
    den = np.sum(U,axis=0)
    for j in range(K):
        U[j,:] = U[j,:]/den
#  cluster at minimum scale
    try:
        U,Ms,Cs,Ps,pdens = em(G,U,T0,beta,rows,cols)
    except:
        print 'em failed' 
        return     
#  sort clusters wrt partition density
    idx = np.argsort(pdens)  
    idx = idx[::-1]
    U = U[idx,:]
#  clustering at increasing scales
    for i in range(max_scale-min_scale):
#      expand U and renormalize         
        U = np.reshape(U,(K,rows,cols))  
        rows = rows*2
        cols = cols*2
        U = ndi.zoom(U,(1,2,2))
        U = np.reshape(U,(K,rows*cols)) 
        idx = np.where(U<0.0)
        U[idx] = 0.0
        den = np.sum(U,axis=0)        
        for j in range(K):
            U[j,:] = U[j,:]/den
#      expand the image
        for i in range(bands):
            DWTbands[i].invert()
        G = np.transpose(np.array([DWTbands[i].get_quadrant(0,float=True).ravel() for i in range(bands)]))  
#      cluster
        unfrozen = np.where(np.max(U,axis=0) < 0.90)
        try:
            U,Ms,Cs,Ps,pdens = em(G,U,0.0,beta,rows,cols,unfrozen=unfrozen)
        except:
            print 'em failed' 
            return                         
    print 'Cluster mean vectors'
    print Ms
    print 'Cluster covariance matrices'
    for k in range(K):
        print 'cluster: %i'%k
        print Cs[k]
#  up-sample class memberships if necessary
    if min_scale>0:
        U = np.reshape(U,(K,rows,cols))
        f = 2**min_scale  
        rows = rows*f
        cols = cols*f
        U = ndi.zoom(U,(1,f,f))
        U = np.reshape(U,(K,rows*cols)) 
        idx = np.where(U<0.0)
        U[idx] = 0.0
        den = np.sum(U,axis=0)        
        for j in range(K):
            U[j,:] = U[j,:]/den        
#  classify
    labels = np.byte(np.argmax(U,axis=0)+1)
    class_image[0:rows,0:cols] = np.reshape(labels,(rows,cols))
    rows1,cols1 = class_image.shape
#  write to disk
    driver = inDataset.GetDriver()    
    outDataset = driver.Create(outfile,cols1,rows1,1,GDT_Byte)
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
#  write class membership probability file if desired  
    if probs:   
        outDataset = driver.Create(probfile,cols,rows,K,GDT_Byte) 
        if geotransform is not None:
            outDataset.SetGeoTransform(tuple(gt)) 
        if projection is not None:
            outDataset.SetProjection(projection)  
        for k in range(K):
            probs = np.reshape(U[k,:],(rows,cols))
            probs = np.byte(probs*255)
            outBand = outDataset.GetRasterBand(k+1)
            outBand.WriteArray(probs,0,0)
            outBand.FlushCache()    
        outDataset = None    
        print 'class probabilities written to: %s'%probfile                                  
    inDataset = None
    if (ext == '') and (K<19):
#  try to make an ENVI classification header file            
        hdr = header.Header() 
        headerfile = outfile+'.hdr'
        f = open(headerfile)
        line = f.readline()
        envihdr = ''
        while line:
            envihdr += line
            line = f.readline()
        f.close()         
        hdr.read(envihdr)
        hdr['file type'] ='ENVI Classification'
        hdr['classes'] = str(K+1)
        classlookup = '{0'
        for i in range(1,3*(K+1)):
            classlookup += ', '+str(str(ctable[i]))
        classlookup +='}'    
        hdr['class lookup'] = classlookup
        hdr['class names'] = ['class %i'%i for i in range(K+1)]
        f = open(headerfile,'w')
        f.write(str(hdr))
        f.close()                 
    print 'classified image written to: '+outfile       
    print 'elapsed time: '+str(time.time()-start)                        
    print '--done------------------------'  
       
if __name__ == '__main__':
    main()    