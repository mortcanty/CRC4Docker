#!/usr/bin/env python
#******************************************************************************
#  Name:     rx.py
#  Purpose:  RX anomaly detection for multi- and hyperspectral images
#  Usage:             
#    python rx.py [options] filename
#
#  Copyright (c) 2018, Mort Canty

import numpy as np
import os, sys, getopt, time
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly,GDT_Float32
from spectral.algorithms.detectors import RX 
from spectral.algorithms.algorithms import calc_stats 
import spectral.io.envi as envi
 
def main():      
    usage = '''
Usage:
------------------------------------------------

RX anomaly detection for multi- and hyperspectral images

python %s [OPTIONS]  filename

Options:
  
  -h         this help
  
-------------------------------------------------'''%sys.argv[0]  
    options,args = getopt.getopt(sys.argv[1:],'h')
    for option, value in options: 
        if option == '-h':
            print usage
            return    
    gdal.AllRegister()
    infile = args[0] 
    path = os.path.dirname(infile)
    basename = os.path.basename(infile)
    root, ext = os.path.splitext(basename)
    outfile = path+'/'+root+'_rx'+ext  
    print '------------ RX ---------------'
    print time.asctime()     
    print 'Input %s'%infile
    start = time.time()        
#  input image, convert to ENVI format                     
    inDataset = gdal.Open(infile,GA_ReadOnly)
    cols = inDataset.RasterXSize
    rows = inDataset.RasterYSize          
    projection = inDataset.GetProjection()
    geotransform = inDataset.GetGeoTransform()       
    driver = gdal.GetDriverByName('ENVI')        
    enviDataset = driver.CreateCopy('imagery/entmp',
                                          inDataset)
    inDataset = None
    enviDataset = None  
#  RX-algorithm        
    img = envi.open('imagery/entmp.hdr')
    arr = img.load()
    rx = RX(background=calc_stats(arr))
    res = rx(arr)
#  output 
    driver = gdal.GetDriverByName('GTiff')    
    outDataset = driver.Create(outfile,cols,rows,1,\
                                    GDT_Float32) 
    if geotransform is not None:
        outDataset.SetGeoTransform(geotransform)
    if projection is not None:
        outDataset.SetProjection(projection) 
    outBand = outDataset.GetRasterBand(1) 
    outBand.WriteArray(np.asarray(res,np.float32),0,0) 
    outBand.FlushCache()
    outDataset = None   
    os.remove('imagery/entmp')
    os.remove('imagery/entmp.hdr')
    print 'Result written to %s'%outfile
    print 'elapsed time: %s'%str(time.time()-start)
    
if __name__ == '__main__':
    main()