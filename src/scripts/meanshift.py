#!/usr/bin/env python
#******************************************************************************
#  Name:     meanshift.py
#  Purpose:  Segment multispectral image with mean shift 
#  Usage:             
#    python meanshift.py 
#
#  Copyright (c) 2018, Mort Canty

import numpy as np
import os, sys, getopt, time
from osgeo import gdal
import auxil.auxil1 as auxil
import scipy.ndimage.filters as filters
from osgeo.gdalconst import GA_ReadOnly, GDT_Float32

def mean_shift(data,idx,hs,nc,nr,nb):
    n = nc*nr
    cpts = np.zeros((nr,nc),dtype=np.int8)
# initialize mean   
    i = idx%nc  # x-position
    j = idx/nc  # y-position
    m  = data[j,i,:] # initial mean
    dm = 100.0
    itr=0
    cpts_max=0  
    while (dm>0) and (itr<100):  
        m1 = m
        bi = max(i-hs,0)
        ei = min(i+hs,nc)
        bj = max(j-hs,0)
        ej = min(j+hs,nr)
        dta = data[bj:ej,bi:ei,:]
        nd = dta.size/(nb+2)
        dta = np.reshape(dta,(nd,nb+2))
        d2 = np.sum((dta-m)**2,1)
        indices = np.where(d2 <= hs**2)[0]
        count = indices.size
        if count > 0:
            ii = indices % (ei-bi)
            jj = indices/(ei-bi)
            cpts_max = max ( cpts_max, min( ((bj+jj)*nc+bi+ii)[count-1]+1, n-1 ) )
#          update mean         
            m = (np.sum(dta[indices,:],0)/count).astype(np.int)
#      flag pixels near the current path
        indices = np.where(d2<=hs**2/9)[0]  
        if indices.size > 0:
            ii = indices%(ei-bi)
            jj = indices/(ei-bi)
            cpts[bj+jj,bi+ii]=1 
        i = m[nb]
        j = m[nb+1]
        dm = np.max(np.abs(m-m1))         
        itr += 1
    return (m,np.reshape(cpts,n),cpts_max)            
               
def main():
 
    usage = '''
Usage: 
--------------------------------------

Segment a multispectral image with mean shift 

python %s [OPTIONS] filename

Options:
  -h            this help
  -p  <list>    band positions e.g. -p [1,2,3,4,5,7]
  -d  <list>    spatial subset [x,y,width,height] 
                              e.g. -d [0,0,200,200]
  -r  <int>     spectral bandwidth (default 15)
  -s  <int>     spatial bandwidth (default 15)
  -m  <int>     minimum segment size (default 30) 

  -------------------------------------'''%sys.argv[0]   
                
    options,args = getopt.getopt(sys.argv[1:],'hs:r:m:d:p:')
    dims = None
    pos = None
    hs = 15
    hr = 15
    minseg = 30
    for option, value in options:
        if option == '-h':
            print usage
            return                
        elif option == '-d':
            dims = eval(value)  
        elif option == '-p':
            pos = eval(value)  
        elif option == '-s':
            hs = eval(value)  
        elif option == '-r':
            hr = eval(value) 
        elif option == '-m':
            minseg = eval(value)   
    gdal.AllRegister()
    infile = args[0]
    inDataset = gdal.Open(infile,GA_ReadOnly)     
    nc = inDataset.RasterXSize
    nr = inDataset.RasterYSize    
    nb = inDataset.RasterCount
    if dims:
        x0,y0,nc,nr = dims
    else:
        x0 = 0
        y0 = 0       
    if pos is not None:
        nb = len(pos)
    else:
        pos = range(1,nb+1)    
    m = nc*nr    
    path = os.path.dirname(infile)
    basename = os.path.basename(infile)
    root, ext = os.path.splitext(basename)
    outfile = path+'/'+root+'_meanshift'+ext        
    print '========================='
    print '    mean shift'
    print '========================='
    print 'infile: %s'%infile 
    start = time.time()                                     
#  input image           
    data = np.zeros((nr,nc,nb+2),dtype=np.float)      
    k = 0
    for b in pos:
        band = inDataset.GetRasterBand(b)
        data[:,:,k] = auxil.bytestr(band.ReadAsArray(x0,y0,nc,nr))
        k += 1
#  normalize spatial/spectral    
    data = data*hs/hr
    ij = np.array(range(nr*nc))  
    data[:,:,nb] = np.reshape(ij%nc,(nr,nc))   # x-coord of (i,j) = j
    data[:,:,nb+1] = np.reshape(ij/nc,(nr,nc)) # y-coord of (i,j) = i
    modes = [np.zeros(nb+2)]
    labeled = np.zeros(m,dtype=np.int)
    idx = 0
    idx_max = 1000
    label = 0
#  loop over all pixels   
    print 'filtering pixels...' 
    while idx<m:    
        mode,cpts,cpts_max = mean_shift(data,idx,hs,nc,nr,nb)
        idx_max = max(idx_max,cpts_max)
#      squared distance to nearest neighbor
        dd = np.sum((mode-modes)**2,1)
        d2 = np.min(dd)
#      label of nearest neighbor
        l_nn = np.argmin(dd)
#      indices of pixels to be labeled
        indices = idx + np.intersect1d( np.where(cpts[idx:idx_max]>0)[0], 
                                        np.where(labeled[idx:idx_max]==0)[0] )
        count = indices.size
        if count>0:
#          label pixels             
            if ((count<minseg) or (d2<hs**2)) and (l_nn!=0):
                labeled[indices]=l_nn 
            else:
                modes = np.append(modes,[mode],axis=0)
                labeled[indices] = label
                label += 1
#          find the next unlabeled pixel
            nxt = idx + np.where(labeled[idx:idx_max]==0)[0]
            count = nxt.size
            if count>0:
                idx = np.min(nxt)
            else:
#              done                
                idx = m
        else:
            idx += 1  
#  write to disk   
    driver = gdal.GetDriverByName('GTiff')    
    outDataset = driver.Create(outfile,nc,nr,nb+2,GDT_Float32)
    projection = inDataset.GetProjection()
    geotransform = inDataset.GetGeoTransform()
    if geotransform is not None:
        gt = list(geotransform)
        gt[0] = gt[0] + x0*gt[1]
        gt[3] = gt[3] + y0*gt[5]
        outDataset.SetGeoTransform(tuple(gt))
    if projection is not None:
        outDataset.SetProjection(projection)    
               
    labeled = filters.median_filter(np.reshape(labeled,(nr,nc)),3)
    boundaries = np.zeros(m)
    
    xx = (labeled-np.roll(labeled,(1,0))).ravel()
    yy = (labeled-np.roll(labeled,(0,1))).ravel()
    idx1 = np.where( xx != 0)[0]
    idx2 = np.where( yy != 0)[0]
    idx = np.union1d(idx1,idx2)        
    boundaries[idx] = 255 
        
    labeled = np.reshape(labeled,m)

    filtered = np.zeros((m,nb))
    labels = modes.shape[0]
    for lbl in range(labels):
        indices = np.where(labeled==lbl)[0]
        filtered[indices,:] = modes[lbl,:nb]        
        
    for k in range(nb):
        outBand = outDataset.GetRasterBand(k+1)
        outBand.WriteArray(np.reshape(filtered[:,k],(nr,nc)),0,0)  
        outBand.FlushCache() 
    outBand = outDataset.GetRasterBand(nb+1)
    outBand.WriteArray(np.reshape(labeled,(nr,nc)),0,0)  
    outBand.FlushCache()    
    outBand = outDataset.GetRasterBand(nb+2)
    outBand.WriteArray(np.reshape(boundaries,(nr,nc)),0,0)  
    outBand.FlushCache()    

    outDataset = None
    inDataset = None
    print 'result written to: '+outfile    
    print 'elapsed time: '+str(time.time()-start)                        
       
if __name__ == '__main__':
    main()    