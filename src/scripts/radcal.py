#!/usr/bin/env python
#******************************************************************************
#  Name:     radcal.py
#  Purpose:  Automatic radiometric normalization
#  Usage:             
#       python radcal.py  [OPTIONS] imadFile [FullSceneFile] 
#
#  Copyright (c) 2018, Mort Canty

import sys, os, time, getopt
import numpy as np
from scipy import stats
from osgeo import gdal
import matplotlib.pyplot as plt
from osgeo.gdalconst import GA_ReadOnly, GDT_Float32
from auxil.auxil1 import orthoregress

 
def main():
    usage = '''
Usage:
------------------------------------------------
Automatic radiometric normalization 
    
python %s [OPTIONS] iMadFile [fullSceneFile]
    
Options:

   -h           this help
   -t  <float>  P-value threshold (default 0.95)
   -d  <list>   spatial subset e.g. -d [0,0,500,500]
   -p  <list>   band positions  e.g. -p [1,2,3]
   
Spatial subset MUST match that of iMadFile
Spectral dimension of fullSceneFile, if present,
MUST match those of target and reference images

iMadFile is assumed to be of form path/MAD(filename1-filename2).ext 
and the output file is named 

         path/filename2_norm.ext.

That is, it is assumed that filename1 is reference and
filename2 is target and the output retains the format
of the imMadFile. A similar convention is used to
name the normalized full scene, if present:

         fullSceneFile_norm.ext

Note that, for ENVI format, ext is the empty string.
   
-------------------------------------------------------'''%sys.argv[0]

    options, args = getopt.getopt(sys.argv[1:],'hnp:d:t:')
    pos = None
    dims = None
    ncpThresh = 0.95  
    fsfn = None  
    graphics = True        
    for option, value in options:
        if option == '-h':
            print usage
            return 
        elif option == '-n':
            graphics = False
        elif option == '-p':
            pos = eval(value)
        elif option == '-d':
            dims = eval(value) 
        elif option == '-t':
            ncpThresh = eval(value)    
    if (len(args) != 1) and (len(args) != 2):
        print 'Incorrect number of arguments'
        print usage
        sys.exit(1)
    imadfn = args[0]
    if len(args) == 2:
        fsfn = args[1]                                    
        path = os.path.dirname(fsfn)
        basename = os.path.basename(fsfn)
        root, ext = os.path.splitext(basename)
        fsoutfn = path+'/'+root+'_norm_all'+ext
    path = os.path.dirname(imadfn)
    basename = os.path.basename(imadfn)
    root, ext = os.path.splitext(basename)
    b = root.find('(')
    e = root.find(')')
    referenceroot, targetbasename = root[b+1:e].split('-')
    referencefn = path + '/' + referenceroot + ext
    targetfn = path + '/' + targetbasename + ext
    targetroot, targetext = os.path.splitext(targetbasename)
    outfn = path + '/' + targetroot + '_norm' + targetext
    imadDataset = gdal.Open(imadfn,GA_ReadOnly)    
    try:
        imadbands = imadDataset.RasterCount 
        cols = imadDataset.RasterXSize
        rows = imadDataset.RasterYSize
    except Exception as e:
        print 'Error: %s  --Image could not be read'%e
        sys.exit(1)
    referenceDataset = gdal.Open(referencefn,GA_ReadOnly)     
    targetDataset = gdal.Open(targetfn,GA_ReadOnly)           
    if pos is None:
        pos = range(1,referenceDataset.RasterCount+1)      
    if dims is None:
        x0 = 0; y0 = 0
    else:
        x0,y0,cols,rows = dims          
    chisqr = imadDataset.GetRasterBand(imadbands).ReadAsArray(0,0,cols,rows).ravel()
    ncp = 1 - stats.chi2.cdf(chisqr,[imadbands-1])
    idx = np.where(ncp>ncpThresh)[0]
    
    m = len(idx)
    
    idx1 = np.random.permutation(m)
    
    idxall = idx[idx1]
    
    idxtrain = idxall[0:2*m//3]
    idxtest = idxall[2*m//3:]
    mtrain = len(idxtrain)
    mtest = len(idxtest)
    
    
    print time.asctime() 
    print 'reference: '+referencefn
    print 'target   : '+targetfn   
    print 'P-value threshold: '+str(ncpThresh)
    print 'no-change pixels for training: %i, for testing: %i' %(mtrain,mtest) 
    start = time.time()           
    driver = targetDataset.GetDriver()    
    outDataset = driver.Create(outfn,cols,rows,len(pos),GDT_Float32)
    projection = imadDataset.GetProjection()
    geotransform = imadDataset.GetGeoTransform()
    if geotransform is not None:
        outDataset.SetGeoTransform(geotransform)
    if projection is not None:
        outDataset.SetProjection(projection)    
    aa = []
    bb = []  
    if graphics:
        plt.figure(1,(10,7))
    j = 1
    bands = len(pos)
    print 'band   slope    intercept  correlation P(t-test)  P(F-test)'
    for k in pos:
        x = referenceDataset.GetRasterBand(k).ReadAsArray(x0,y0,cols,rows).astype(float).ravel()
        y = targetDataset.GetRasterBand(k).ReadAsArray(x0,y0,cols,rows).astype(float).ravel()
        b,a,R = orthoregress(y[idxtrain],x[idxtrain])
        my = max(y[idxtrain])
        if (j<7) and graphics:
            plt.subplot(2,3,j)
            plt.plot(y[idxtrain],x[idxtrain],'.')
            plt.plot([0,my],[a,a+b*my])
            plt.title('Band %i'%k)
            if ((j<4) and (bands<4)) or j>3:
                plt.xlabel('Target')
            if (j==1) or (j==4):
                plt.ylabel('Reference')
        aa.append(a)
        bb.append(b)     
        outBand = outDataset.GetRasterBand(j)       
        _,Pt = stats.ttest_ind(x[idxtest],a+b*y[idxtest])
        f = np.var(x[idxtest])/np.var(a+b*y[idxtest])  
        if f < 1.0:
            f = 1/f   
        Pf = stats.f.sf(f,mtest-1,mtest-1)      
        print ' %i   %f   %f   %f   %f   %f' %(k,b,a,R,Pt,Pf)       
        outBand.WriteArray(np.resize(a+b*y,(rows,cols)),0,0) 
        outBand.FlushCache()
        j += 1
    if graphics:
        plt.show() 
        plt.close()   
    referenceDataset = None
    targetDataset = None
    outDataset = None
    print 'result written to: '+outfn 
    if fsfn is not None:
        print 'normalizing '+fsfn+'...'
        fsDataset = gdal.Open(fsfn,GA_ReadOnly)
        try:
            cols = fsDataset.RasterXSize
            rows = fsDataset.RasterYSize    
        except Exception as e:
            print 'Error %s  -- Image could not be read in' 
            sys.exit(1)   
        driver = fsDataset.GetDriver()
        outDataset = driver.Create(fsoutfn,cols,rows,len(pos),GDT_Float32)
        projection = fsDataset.GetProjection()
        geotransform = fsDataset.GetGeoTransform()
        if geotransform is not None:
            outDataset.SetGeoTransform(geotransform)
        if projection is not None:
            outDataset.SetProjection(projection) 
        j = 1    
        for k in pos:
            inBand = fsDataset.GetRasterBand(k)
            outBand = outDataset.GetRasterBand(j)
            for i in range(rows):
                y = inBand.ReadAsArray(0,i,cols,1)
                outBand.WriteArray(aa[j-1]+bb[j-1]*y,0,i) 
            outBand.FlushCache()  
            j += 1     
        outDataset = None   
        fsDataset = None 
        print 'full result written to: '+fsoutfn   
    print 'elapsed time: %s'%str(time.time()-start)
    
if __name__ == '__main__':
    main()        
