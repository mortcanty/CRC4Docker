#!/usr/bin/env python
#******************************************************************************
#  Name:     dwt.py
#  Purpose:  Panchromatic sharpening with the discrete wavelet transform 
#  Usage:             
#    python dwt.py 
#
#  Copyright (c) 2018, Mort Canty

import auxil.auxil1 as auxil
import os, sys, getopt, time, copy
import numpy as np
from osgeo import gdal
import scipy.ndimage.interpolation as ndii
from osgeo.gdalconst import GA_ReadOnly, GDT_Float32      

def main():
    usage = '''
Usage: 
--------------------------------------

Panchromatic sharpening with the a discrete transform

python %s [OPTIONS] msfilename panfilename 

Options:
  -h            this help
  -p  <list>    RGB band positions to be sharpened (default all)
                               e.g. -p [1,2,3]
  -d  <list>    spatial subset [x,y,width,height] of ms image
                               e.g. -d [0,0,200,200]
  -r  <int>     resolution ratio ms:pan (default 4)
  -b  <int>     ms band for co-registration 
  
  -------------------------------------'''%sys.argv[0]
  
    options, args = getopt.getopt(sys.argv[1:],'hd:p:r:b:')
    ratio = 4
    dims1 = None
    pos1 = None  
    k1 = 0          
    for option, value in options:
        if option == '-h':
            print usage
            return 
        elif option == '-r':
            ratio = eval(value)
        elif option == '-d':
            dims1 = eval(value) 
        elif option == '-p':
            pos1 = eval(value)    
        elif option == '-b':
            k1 = eval(value)-1
    if len(args) != 2:
        print 'Incorrect number of arguments'
        print usage
        sys.exit(1)                         
    gdal.AllRegister()
    file1 = args[0]
    file2 = args[1]   
    path = os.path.dirname(file1)
    basename1 = os.path.basename(file1)
    root1, ext1 = os.path.splitext(basename1)
    outfile = '%s/%s_pan_dwt%s'%(path,root1,ext1)       
#  MS image    
    inDataset1 = gdal.Open(file1,GA_ReadOnly)     
    try:    
        cols = inDataset1.RasterXSize
        rows = inDataset1.RasterYSize    
        bands = inDataset1.RasterCount
    except Exception as e:
        print 'Error: %e --Image could not be read'%e
        sys.exit(1)    
    if pos1 is None:
        pos1 = range(1,bands+1)
    num_bands = len(pos1)    
    if dims1 is None:
        dims1 = [0,0,cols,rows]
    x10,y10,cols1,rows1 = dims1    
#  PAN image    
    inDataset2 = gdal.Open(file2,GA_ReadOnly)     
    try:  
        bands = inDataset2.RasterCount
    except Exception as e:
        print 'Error: %e --Image could not be read'%e  
        sys.exit(1)   
    if bands>1:
        print 'PAN image must be a single band'
        sys.exit(1)     
    geotransform1 = inDataset1.GetGeoTransform()
    geotransform2 = inDataset2.GetGeoTransform()   
    if (geotransform1 is None) or (geotransform2 is None):
        print 'Image not georeferenced, aborting' 
        sys.exit(1)      
    print '========================='
    print '   DWT Pansharpening'
    print '========================='
    print time.asctime()     
    print 'MS  file: '+file1
    print 'PAN file: '+file2       
#  image arrays
    band = inDataset1.GetRasterBand(1)
    tmp = band.ReadAsArray(0,0,1,1)
    dt = tmp.dtype
    MS = np.asarray(np.zeros((num_bands,rows1,cols1)),dtype=dt) 
    k = 0                                   
    for b in pos1:
        band = inDataset1.GetRasterBand(b)
        MS[k,:,:] = band.ReadAsArray(x10,y10,cols1,rows1)
        k += 1
#  if integer assume 11bit quantization otherwise must be byte   
    if MS.dtype == np.int16:
        fact = 8.0
        MS = auxil.bytestr(MS,(0,2**11)) 
    else:
        fact = 1.0
#  read in corresponding spatial subset of PAN image    
    if (geotransform1 is None) or (geotransform2 is None):
        print 'Image not georeferenced, aborting' 
        return
#  upper left corner pixel in PAN    
    gt1 = list(geotransform1)               
    gt2 = list(geotransform2)
    ulx1 = gt1[0] + x10*gt1[1]
    uly1 = gt1[3] + y10*gt1[5]
    x20 = int(round(((ulx1 - gt2[0])/gt2[1])))
    y20 = int(round(((uly1 - gt2[3])/gt2[5])))
    cols2 = cols1*ratio
    rows2 = rows1*ratio
    band = inDataset2.GetRasterBand(1)
    PAN = band.ReadAsArray(x20,y20,cols2,rows2)        
#  if integer assume 11-bit quantization, otherwise must be byte    
    if PAN.dtype == np.int16:
        PAN = auxil.bytestr(PAN,(0,2**11))                                   
#  compress PAN to resolution of MS image  
    panDWT = auxil.DWTArray(PAN,cols2,rows2)          
    r = ratio
    while r > 1:
        panDWT.filter()
        r /= 2
    bn0 = panDWT.get_quadrant(0) 
    lines0,samples0 = bn0.shape    
    bn1 = MS[k1,:,:]  
#  register (and subset) MS image to compressed PAN image 
    (scale,angle,shift) = auxil.similarity(bn0,bn1)
    tmp = np.zeros((num_bands,lines0,samples0))
    for k in range(num_bands): 
        bn1 = MS[k,:,:]                    
        bn2 = ndii.zoom(bn1, 1.0/scale)
        bn2 = ndii.rotate(bn2, angle)
        bn2 = ndii.shift(bn2, shift)
        tmp[k,:,:] = bn2[0:lines0,0:samples0]        
    MS = tmp            
#  compress pan once more, extract wavelet quadrants, and restore
    panDWT.filter()  
    fgpan = panDWT.get_quadrant(1)
    gfpan = panDWT.get_quadrant(2)
    ggpan = panDWT.get_quadrant(3)    
    panDWT.invert()       
#  output array            
    sharpened = np.zeros((num_bands,rows2,cols2),dtype=np.float32)     
    aa = np.zeros(3)
    bb = np.zeros(3)       
    print 'Wavelet correlations:'                                   
    for i in range(num_bands):
#      make copy of panDWT and inject ith ms band                
        msDWT = copy.deepcopy(panDWT)
        msDWT.put_quadrant(MS[i,:,:],0)
#      compress once more                 
        msDWT.filter()
#      determine wavelet normalization coefficents                
        ms = msDWT.get_quadrant(1)    
        aa[0],bb[0],R = auxil.orthoregress(fgpan.ravel(), ms.ravel())
        Rs = 'Band '+str(i+1)+': %8.3f'%R
        ms = msDWT.get_quadrant(2)
        aa[1],bb[1],R = auxil.orthoregress(gfpan.ravel(), ms.ravel())
        Rs += '%8.3f'%R                     
        ms = msDWT.get_quadrant(3)
        aa[2],bb[2],R = auxil.orthoregress(ggpan.ravel(), ms.ravel()) 
        Rs += '%8.3f'%R    
        print Rs         
#      restore once and normalize wavelet coefficients
        msDWT.invert() 
        msDWT.normalize(aa,bb)   
#      restore completely and collect result
        r = 1
        while r < ratio:
            msDWT.invert()
            r *= 2                            
        sharpened[i,:,:] = msDWT.get_quadrant(0)      
    sharpened *= fact    
#  write to disk       
    driver = inDataset1.GetDriver()
    outDataset = driver.Create(outfile,cols2,rows2,num_bands,GDT_Float32)
    projection1 = inDataset1.GetProjection()
    if projection1 is not None:
        outDataset.SetProjection(projection1)        
    gt1 = list(geotransform1)
    gt1[0] += x10*ratio  
    gt1[3] -= y10*ratio
    gt1[1] = gt2[1]
    gt1[2] = gt2[2]
    gt1[4] = gt2[4]
    gt1[5] = gt2[5]
    outDataset.SetGeoTransform(tuple(gt1))   
    for k in range(num_bands):        
        outBand = outDataset.GetRasterBand(k+1)
        outBand.WriteArray(sharpened[k,:,:],0,0) 
        outBand.FlushCache() 
    outDataset = None    
    print 'Result written to %s'%outfile    
    inDataset1 = None
    inDataset2 = None                      

if __name__ == '__main__':
    main()