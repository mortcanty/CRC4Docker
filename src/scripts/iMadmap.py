#!/usr/bin/env python
#******************************************************************************
#  Name:     iMadmap.py
#  Purpose: Make change map from iMAD variates
#  Usage (from command line):             
#    python iMadmap.py  [options] fileNmae significance
#
#  Copyright (c) 2018 Mort Canty

import numpy as np
import os, sys, getopt
from scipy import stats, ndimage
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly,GDT_Float32

def main():    
    usage = '''
Usage:
------------------------------------------------
Make a change map from iMAD variates at a given significance level 
    
python %s [OPTIONS] madfile significance
    
Options:

   -h           this help
   -m           run a 3x3 median filter over the P-values
   -d  <list>   spatial subset list e.g. -d [0,0,500,500]
   
-----------------------------------------------------''' %sys.argv[0]
                       
    options,args = getopt.getopt(sys.argv[1:],'hmd:')
    dims = None
    pos = None
    median = False
    for option, value in options: 
        if option == '-h':
            print usage
            return        
        elif option == '-m':
            median = True   
        elif option == '-d':
            dims = eval(value)  
    if len(args) != 2:
        print 'Incorrect number of arguments'
        print usage
        return       
    gdal.AllRegister()
    infile = args[0] 
    alpha = eval(args[1])
    path = os.path.dirname(infile)
    basename = os.path.basename(infile)
    root, ext = os.path.splitext(basename)
    outfile = path+'/'+root+'_cmap'+ext  
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
    if pos is None:
        pos = [1,2,3]  
#  data matrix for MADs
    mads= np.zeros((rows*cols,bands-1))                               
    for b in range(bands-1):
        band = inDataset.GetRasterBand(b+1)
        mads[:,b] = band.ReadAsArray(x0,y0,cols,rows).astype(float).ravel()   
    band = inDataset.GetRasterBand(bands)
    chisqr =  band.ReadAsArray(x0,y0,cols,rows).astype(float).ravel()  
    P = 1-stats.chi2.cdf(chisqr,[bands-1])
    if median:
        P = ndimage.filters.median_filter(np.reshape(P,(rows,cols)), size = (3,3))
        P = np.reshape(P,rows*cols)
    idx = np.where(P>alpha)[0]
    mads[idx,:] = 0.0
    mads = np.reshape(mads,(rows,cols,bands-1))
#  write to disk       
    driver = inDataset.GetDriver() 
    outDataset = driver.Create(outfile,
                cols,rows,bands-1,GDT_Float32)
    projection = inDataset.GetProjection()
    geotransform = inDataset.GetGeoTransform()
    if geotransform is not None:
        gt = list(geotransform)
        gt[0] = gt[0] + x0*gt[1]
        gt[3] = gt[3] + y0*gt[5]
        outDataset.SetGeoTransform(tuple(gt))
    if projection is not None:
        outDataset.SetProjection(projection)        
    for k in range(bands-1):        
        outBand = outDataset.GetRasterBand(k+1)
        outBand.WriteArray(mads[:,:,k],0,0) 
        outBand.FlushCache() 
    print 'change map written to: %s'%outfile    
     
if __name__ == '__main__':
    main()    