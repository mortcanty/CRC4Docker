#!/usr/bin/env python
#******************************************************************************
#  Name:     registerms.py
#  Purpose:  
#    Perform image-image registration of two optical/infrared images 
#    via similarity warping.
#
#  Usage:     
#    from auxil import registerms
#    registerms.register(reffilename,warpfilename,dims,outfile) 
#          or        
#    python registerms.py [OPTIONS] reffilename warpfilename
#
#  Copyright (c) 2018 Mort Canty

from auxil.auxil1 import similarity
import os, sys, getopt, time
import numpy as np
from osgeo import gdal
import scipy.ndimage.interpolation as ndii
from osgeo.gdalconst import GA_ReadOnly, GDT_Float32
  
def register(fn1, fn2, warpband, dims1=None, outfile=None):                  
    gdal.AllRegister()    
    print '--------------------------------'
    print'        Register'   
    print'---------------------------------'      
    print time.asctime()     
    print 'reference image: '+fn1
    print 'warp image: '+fn2     
    print 'warp band: %i'%warpband  
    
    start =  time.time()              
    try:
        if outfile is None:
            path2 = os.path.dirname(fn2)
            basename2 = os.path.basename(fn2)
            root2, ext2 = os.path.splitext(basename2)
            outfile = path2 + '/' + root2 + '_warp' + ext2
        inDataset1 = gdal.Open(fn1,GA_ReadOnly)     
        inDataset2 = gdal.Open(fn2,GA_ReadOnly)
        try:
            cols1 = inDataset1.RasterXSize
            rows1 = inDataset1.RasterYSize    
            cols2 = inDataset2.RasterXSize
            rows2 = inDataset2.RasterYSize    
            bands2 = inDataset2.RasterCount   
        except Exception as e:
            print 'Error %s  --Image could not be read in'%e
            sys.exit(1)     
        if dims1 is None:
            x0 = 0
            y0 = 0
        else:
            x0,y0,cols1,rows1 = dims1    
        
        band = inDataset1.GetRasterBand(warpband)
        refband = band.ReadAsArray(x0,y0,cols1,rows1).astype(np.float32)
        band = inDataset2.GetRasterBand(warpband)
        warpband = band.ReadAsArray(x0,y0,cols1,rows1).astype(np.float32)
        
    #  similarity transform parameters for reference band number            
        scale, angle, shift = similarity(refband, warpband)
    
        driver = inDataset2.GetDriver()
        outDataset = driver.Create(outfile,cols1,rows1,bands2,GDT_Float32)
        projection = inDataset1.GetProjection()
        geotransform = inDataset1.GetGeoTransform()
        if geotransform is not None:
            gt = list(geotransform)
            gt[0] = gt[0] + x0*gt[1]
            gt[3] = gt[3] + y0*gt[5]
            outDataset.SetGeoTransform(tuple(gt))
        if projection is not None:
            outDataset.SetProjection(projection) 
    
    #  warp 
        for k in range(bands2):       
            inband = inDataset2.GetRasterBand(k+1)      
            outBand = outDataset.GetRasterBand(k+1)
            bn1 = inband.ReadAsArray(0,0,cols2,rows2).astype(np.float32)
            bn2 = ndii.zoom(bn1, 1.0 / scale)
            bn2 = ndii.rotate(bn2, angle)
            bn2 = ndii.shift(bn2, shift)       
            outBand.WriteArray(bn2[y0:y0+rows1, x0:x0+cols1]) 
            outBand.FlushCache() 
        inDataset1 = None
        inDataset2 = None
        outDataset = None    
        print 'Warped image written to: %s'%outfile
        print 'elapsed time: %s'%str(time.time()-start)
        return outfile
    except Exception as e:
        print 'registersms failed: %s'%e    
        return None   
    
def main(): 
    usage = '''
    Usage:
------------------------------------------------

python %s [OPTIONS] reffilename warpfilename
    
Perform image-image registration of two polarimetric SAR images   
    
Options:

   -h         this help
   -d  <list> spatial subset list e.g. -d [0,0,500,500]
   -b  <int>  band to use for warping (default 1)

Choose a reference image, the image to be warped and, optionally,
the band to be used for warping (default band 1) and the spatial subset
of the reference image. 

The reference image should be smaller than the warp image 
(i.e., the warp image should overlap the reference image completely) 
and its upper left corner should be near that of the warp image:
----------------------
|   warp image
|
|  --------------------
|  |
|  |  reference image
|  |   

The reference image (or spatial subset) should not contain zero data

The warped image (warpfile_warp) will be trimmed to the spatial 
dimensions of the reference image.
------------------------------------------------''' %sys.argv[0]
    options, args = getopt.getopt(sys.argv[1:],'hb:d:')  
    warpband = 1
    dims1 = None
    for option, value in options:
        if option == '-h':
            print usage
            return   
        elif option == '-b':
            warpband = eval(value)      
        elif option == '-d':
            dims1 = eval(value)    
    if len(args) != 2:
        print 'Incorrect number of arguments'
        print usage
        sys.exit(1)      
    fn1 = args[0]  # reference
    fn2 = args[1]  # warp  
    outfile = register(fn1,fn2,warpband,dims1)   
   
if __name__ == '__main__':
    main()    