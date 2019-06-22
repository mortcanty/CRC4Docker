#!/usr/bin/env python
#******************************************************************************
#  Name:     sar_seqQ.py
#  Purpose:  Perform sequential change detection on multi-temporal, polarimetric SAR imagery 
#            Determine time(s) at which change occurred, see
#            Condradsen et al. (2016) IEEE Transactions on Geoscience and Remote Sensing,
#            Vol. 54 No. 5 pp. 3007-3024
#            Tests based both upon Rj and Q = Prod(Rj)   
#
#  Usage:             
#    python sar_seqQ.py [OPTIONS] filenamelist enl
#
# MIT License
# 
# Copyright (c) 2018 Mort Canty

def call_register((fn0,fni,dims)):
    from auxil.registersar import register
    return register(fn0,fni,dims)

def call_median_filter(pv):
    from scipy import ndimage
    return ndimage.filters.median_filter(pv, size = (3,3))
 
def PV((fns,n,cols,rows,bands)):
    '''Return p-values for change indices R^ell_j'''        
    import numpy as np
    from osgeo.gdalconst import GA_ReadOnly
    from osgeo import gdal
    import sys
    from scipy import stats  
    def getmat(fn,cols,rows,bands):
    #  read 9- 4- 3- 2- or 1-band preprocessed polarimetric matrix files 
    #  and return (complex) matrix elements
        try:
            inDataset1 = gdal.Open(fn,GA_ReadOnly)     
            if bands == 9:
        #      T11 (k1)
                b = inDataset1.GetRasterBand(1)
                k1 = b.ReadAsArray(0,0,cols,rows)
        #      T12  (a1)
                b = inDataset1.GetRasterBand(2)
                a1 = b.ReadAsArray(0,0,cols,rows)
                b = inDataset1.GetRasterBand(3)    
                im = b.ReadAsArray(0,0,cols,rows)
                a1 = (a1 + 1j*im)
        #      T13  (rho1)
                b = inDataset1.GetRasterBand(4)
                rho1 = b.ReadAsArray(0,0,cols,rows)
                b = inDataset1.GetRasterBand(5)
                im = b.ReadAsArray(0,0,cols,rows)
                rho1 = (rho1 + 1j*im)      
        #      T22 (xsi1)
                b = inDataset1.GetRasterBand(6)
                xsi1 = b.ReadAsArray(0,0,cols,rows)    
        #      T23 (b1)        
                b = inDataset1.GetRasterBand(7)
                b1 = b.ReadAsArray(0,0,cols,rows)
                b = inDataset1.GetRasterBand(8)
                im = b.ReadAsArray(0,0,cols,rows)
                b1 = (b1 + 1j*im)      
        #      T33 (zeta1)
                b = inDataset1.GetRasterBand(9)
                zeta1 = b.ReadAsArray(0,0,cols,rows) 
                result = (k1,a1,rho1,xsi1,b1,zeta1)             
            elif bands == 4:
        #      C11 (k1)
                b = inDataset1.GetRasterBand(1)
                k1 = b.ReadAsArray(0,0,cols,rows)
        #      C12  (a1)
                b = inDataset1.GetRasterBand(2)
                a1 = b.ReadAsArray(0,0,cols,rows)
                b = inDataset1.GetRasterBand(3)
                im = b.ReadAsArray(0,0,cols,rows)
                a1 = (a1 + 1j*im)        
        #      C22 (xsi1)
                b = inDataset1.GetRasterBand(4)
                xsi1 = b.ReadAsArray(0,0,cols,rows)   
                result = (k1,a1,xsi1)
            elif bands == 3:
        #      T11 (k1)
                b = inDataset1.GetRasterBand(1)
                k1 = b.ReadAsArray(0,0,cols,rows)
        #      T22 (xsi1)
                b = inDataset1.GetRasterBand(2)
                xsi1 = b.ReadAsArray(0,0,cols,rows)    
        #      T33 (zeta1)
                b = inDataset1.GetRasterBand(3)
                zeta1 = b.ReadAsArray(0,0,cols,rows) 
                result = (k1,xsi1,zeta1)          
            elif bands == 2:
        #      C11 (k1)
                b = inDataset1.GetRasterBand(1)
                k1 = b.ReadAsArray(0,0,cols,rows)    
        #      C22 (xsi1)
                b = inDataset1.GetRasterBand(2)
                xsi1 = b.ReadAsArray(0,0,cols,rows)  
                result = (k1,xsi1)         
            elif bands == 1:        
        #      C11 (k1)
                b = inDataset1.GetRasterBand(1)
                k1 = b.ReadAsArray(0,0,cols,rows) 
                result = (k1,)
            inDataset1 = None
            return result
        except Exception as e:
            print 'Error: %s  -- Could not read file'%e
            sys.exit(1)   
    
    j = np.float64(len(fns))
    eps = sys.float_info.min
    k = 0.0; a = 0.0; rho = 0.0; xsi = 0.0; b = 0.0; zeta = 0.0
    for fn in fns:
        result = getmat(fn,cols,rows,bands)
        if bands==9:
            k1,a1,rho1,xsi1,b1,zeta1 = result
            k1 = n*np.float64(k1)
            a1 = n*np.complex128(a1)
            rho1 = n*np.complex128(rho1)
            xsi1 = n*np.float64(xsi1)
            b1 = n*np.complex128(b1)
            zeta1 = n*np.float64(zeta1)
            k += k1; a += a1; rho += rho1; xsi += xsi1; b += b1; zeta += zeta1  
        elif bands==4:
            k1,a1,xsi1 = result
            k1 = n*np.float64(k1)
            a1 = n*np.complex128(a1)
            xsi1 = n*np.float64(xsi1)
            k += k1; a += a1; xsi += xsi1
        elif bands==3:
            k1,xsi1,zeta1 = result
            k1 = n*np.float64(k1)
            xsi1 = n*np.float64(xsi1)
            zeta1 = n*np.float64(zeta1)
            k += k1; xsi += xsi1; zeta += zeta1  
        elif bands==2:
            k1,xsi1 = result
            k1 = n*np.float64(k1)
            xsi1 = n*np.float64(xsi1)
            k += k1; xsi += xsi1
        elif bands==1:
            k1 = n*np.float64(result[0])
            k += k1              
    if bands==9: 
        p = 3
        detsumj = k*xsi*zeta + 2*np.real(a*b*np.conj(rho)) - xsi*(abs(rho)**2) - k*(abs(b)**2) - zeta*(abs(a)**2) 
        k -= k1; a -= a1; rho -= rho1; xsi -= xsi1; b -= b1; zeta -= zeta1 
        detsumj1 = k*xsi*zeta + 2*np.real(a*b*np.conj(rho)) - xsi*(abs(rho)**2) - k*(abs(b)**2) - zeta*(abs(a)**2)
        detj = k1*xsi1*zeta1 + 2*np.real(a1*b1*np.conj(rho1)) - xsi1*(abs(rho1)**2) - k1*(abs(b1)**2) - zeta1*(abs(a1)**2)
    elif bands==4:
        p = 2
        detsumj = k*xsi - abs(a)**2 
        k -= k1; a -= a1; xsi -= xsi1
        detsumj1 = k*xsi - abs(a)**2
        detj = k1*xsi1 - abs(a1)**2
    elif bands==3:
        p = 3
        detsumj = k*xsi*zeta 
        k -= k1; xsi -= xsi1;  zeta -= zeta1 
        detsumj1 = k*xsi*zeta 
        detj = k1*xsi1*zeta1
    elif bands==2:
        p = 2
        detsumj = k*xsi
        k -= k1; xsi -= xsi1
        detsumj1 = k*xsi
        detj = k1*xsi1
    elif bands==1:
        p = 1
        detsumj = k+0.0 # !!! deep copy
        k -= k1
        detsumj1 = k
        detj = k1    
    detsumj = np.nan_to_num(detsumj)    
    detsumj = np.where(detsumj <= eps,eps,detsumj)
    logdetsumj = np.log(detsumj)
    detsumj1 = np.nan_to_num(detsumj1)
    detsumj1 = np.where(detsumj1 <= eps,eps,detsumj1)
    logdetsumj1 = np.log(detsumj1)
    detj = np.nan_to_num(detj)
    detj = np.where(detj <= eps,eps,detj)
    logdetj = np.log(detj)
#  test statistic
    lnRj = n*( p*( j*np.log(j)-(j-1)*np.log(j-1.) ) + (j-1)*logdetsumj1 + logdetj - j*logdetsumj )  
    if (bands==9) or (bands==4) or (bands==1):
#      full quad, dual pol or intensity (p = 3, 2 or 1)      
        f =p**2
    else:
#      quad and dual diagonal matrix cases (f = 3 or 2, p1 = p2 (= p3) =: p = 1)
        f = bands
        p = 1
    rhoj = 1 - (2.*p**2 - 1)*(1. + 1./(j*(j-1)))/(6.*p*n)
    omega2j = -(f/4.)*(1.-1./rhoj)**2 + (1./(24.*n*n))*p*p*(p*p-1)*(1+(2.*j-1)/(j*(j-1))**2)/rhoj**2     
#  return (p-values, lnRj)  
    Z = -2*rhoj*lnRj
    return ( 1.0-((1.-omega2j)*stats.chi2.cdf(Z,[f])+omega2j*stats.chi2.cdf(Z,[f+4])), lnRj )

def change_maps(pvarray,significance):
    import numpy as np
    k = pvarray.shape[0]
    n = pvarray.shape[2]
#  map of most recent change occurrences
    cmap = np.zeros(n,dtype=np.byte)    
#  map of first change occurrence
    smap = np.zeros(n,dtype=np.byte)
#  change frequency map 
    fmap = np.zeros(n,dtype=np.byte)
#  bitemporal change maps
    bmap = np.zeros((n,k-1),dtype=np.byte)  
    for ell in range(k-1):        
        pvQ = pvarray[ell,k-1,:]          
        for j in range(ell,k-1):
            pv = pvarray[ell,j,:]
            idx = np.where((pv<=significance)&(pvQ<=significance)&(cmap==ell))
            fmap[idx] += 1 
            cmap[idx] = j+1 
            bmap[idx,j] = 255 
            if ell==0:
                smap[idx] = j+1    
    return (cmap,smap,fmap,bmap)

def getpvQ(lnQ,bands,k,n):
    import math
    from scipy import stats
#  test statistic 
    if (bands==9) or (bands==4) or (bands==1):
#      full quad, dual pol or intensity (p = 3, 2 or 1)   
        p = math.sqrt(bands)   
        f =(k-1)*p**2
        rho = 1.0 - (2*p**2-1)*(k/n-1.0/(n*k))/(6.0*(k-1)*p)
        omega2 = p**2*(p**2-1)*(k/n**2 - 1.0/(n*n*k*k))/(24.0*rho**2) - p**2*(k-1)*(1.0-1.0/rho)**2/4.0
    elif bands==2 or bands==3:
#      quad and dual diagonal matrix cases, use first order approx 
        f = (k-1)*bands 
        rho = 1.0
        omega2 = 0.0       
#  return p-value  
    Z = -2*rho*lnQ
    return 1.0-((1.-omega2)*stats.chi2.cdf(Z,[f])+omega2*stats.chi2.cdf(Z,[f+4]))   
                       
def main():  
    import numpy as np
    import os, sys, time, getopt
    from osgeo import gdal
    from auxil import subset
    from ipyparallel import Client
    from osgeo.gdalconst import GA_ReadOnly, GDT_Byte
    from tempfile import NamedTemporaryFile
    usage = '''
Usage:
------------------------------------------------

Sequential change detection for polarimetric SAR images

python %s [OPTIONS]  infiles* outfile enl

Options:
  
  -h           this help 
  -d  <list>   files are to be co-registered to a subset dims = [x0,y0,rows,cols] of the first image, otherwise
               it is assumed that the images are co-registered and have identical spatial dimensions  
  -m           run 3x3 median filter over p-values   
  -s  <float>  significance level for change detection (default 0.0001)

infiles:

  full paths to all input files: /path/to/infile_1 /path/to/infile_1 ... /path/to/infile_k
  
outfile:

  without path (will be written to same directory as infile_1)
  
enl:

  equivalent number of looks

-------------------------------------------------'''%sys.argv[0]

    options,args = getopt.getopt(sys.argv[1:],'hmd:s:')
    dims = None
    significance = 0.0001
    medianfilter = False
    for option, value in options: 
        if option == '-h':
            print usage
            return 
        elif option == '-m':
            medianfilter = True
        elif option == '-d':
            dims = eval(value)
        elif option == '-s':
            significance = eval(value)   
    k = len(args)-2
    fns = args[0:k]  
    n = np.float64(eval(args[-1])) 
    outfn = args[-2]
    gdal.AllRegister()   
    start = time.time()    
#  first SAR image   
    try:            
        inDataset1 = gdal.Open(fns[0],GA_ReadOnly)                             
        cols = inDataset1.RasterXSize
        rows = inDataset1.RasterYSize    
        bands = inDataset1.RasterCount
    except Exception as e:
        print 'Error: %s  -- Could not read file'%e
        sys.exit(1)    
    if dims is not None:
#  images are not yet co-registered, so subset first image and register the others
        _,_,cols,rows = dims
        fn0 = subset.subset(fns[0],dims)
        args1 = [(fns[0],fns[i],dims) for i in range(1,k)]
        try:
            print ' \nattempting parallel execution of co-registration ...' 
            start1 = time.time()  
            c = Client()
            print 'available engines %s'%str(c.ids)
            v = c[:]  
            v.execute('from registersar import register') 
            fns = v.map_sync(call_register,args1)
            print 'elapsed time for co-registration: '+str(time.time()-start1) 
        except Exception as e: 
            start1 = time.time()
            print '%s \nFailed, so running sequential co-registration ...'%e
            fns = map(call_register,args1)  
            print 'elapsed time for co-registration: '+str(time.time()-start)
        fns.insert(0,fn0)  
#      point inDataset1 to the subset image for correct georefrerencing         
        inDataset1 = gdal.Open(fn0,GA_ReadOnly)           
    print '==============================================='
    print '     Multi-temporal SAR Change Detection'
    print '==============================================='   
    print time.asctime()  
    print 'First (reference) filename:  %s'%fns[0]
    print 'number of images: %i'%k
    print 'equivalent number of looks: %f'%n
    print 'significance level: %f'%significance
    if bands==9:
        print 'Quad ploarization'
    elif bands==4:
        print 'Dual polarizaton'
    elif bands==3:
        print 'Quad polarization, diagonal only'
    elif bands==2:
        print 'Dual polarization, diagonal only'
    else:
        print 'Intensity image'
#  output file
    path = os.path.abspath(fns[0])    
    dirn = os.path.dirname(path)
    outfn = dirn + '/' + outfn 
#  create temporary, memory-mapped array of change indices p(Ri<ri)
    mm = NamedTemporaryFile()
    pvarray = np.memmap(mm.name,dtype=np.float64,mode='w+',shape=(k,k,rows*cols))  
    print 'pre-calculating Rj and p-values ...' 
    start1 = time.time() 
    try:
        print 'attempting parallel calculation ...' 
        c = Client()
        print 'available engines %s'%str(c.ids)
        v = c[:]   
        print 'ell = ',
        sys.stdout.flush()      
        for i in range(k-1):  
            print i+1,  
            sys.stdout.flush()              
            args1 = [(fns[i:j+2],n,cols,rows,bands) for j in range(i,k-1)]         
            results = v.map_sync(PV,args1) # list of tuples (p-value, lnRj)
            pvs = [result[0] for result in results] 
            if medianfilter:
                pvs = v.map_sync(call_median_filter,pvs)
            lnRjs = np.array([result[1] for result in results]) 
            lnQ = np.sum(lnRjs,axis=0)            
            pvQ = getpvQ(lnQ,bands,k-i,n)       
            for j in range(i,k-1):
                pvarray[i,j,:] = pvs[j-i].ravel() 
            pvarray[i,k-1,:] = pvQ.ravel()    
    except Exception as e: 
        print '%s \nfailed, so running sequential calculation ...'%e  
        print 'ell= ',
        sys.stdout.flush()  
        for i in range(k-1):        
            print i+1,   
            sys.stdout.flush()             
            args1 = [(fns[i:j+2],n,cols,rows,bands) for j in range(i,k-1)]                         
            results = map(PV,args1)  # list of tuples (p-value, lnRj)
            pvs = [result[0] for result in results] 
            if medianfilter:
                pvs = map(call_median_filter,pvs) 
            lnRjs = np.array([result[1] for result in results]) 
            lnQ = np.sum(lnRjs,axis=0)              
            pvQ = getpvQ(lnQ,bands,k-i,n)                   
            for j in range(i,k-1):
                pvarray[i,j,:] = pvs[j-i].ravel() 
            pvarray[i,k-1,:] = pvQ.ravel()  
    print '\nelapsed time for p-value calculation: '+str(time.time()-start1)    
    cmap,smap,fmap,bmap = change_maps(pvarray,significance)
#  write to file system    
    cmap = np.reshape(cmap,(rows,cols))
    fmap = np.reshape(fmap,(rows,cols))
    smap = np.reshape(smap,(rows,cols))
    bmap = np.reshape(bmap,(rows,cols,k-1))
    driver = inDataset1.GetDriver() 
    basename = os.path.basename(outfn)
    name, _ = os.path.splitext(basename)
    outfn1=outfn.replace(name,name+'_cmap')
    outDataset = driver.Create(outfn1,cols,rows,1,GDT_Byte)
    geotransform = inDataset1.GetGeoTransform()
    if geotransform is not None:
        outDataset.SetGeoTransform(geotransform)
    projection = inDataset1.GetProjection()        
    if projection is not None:
        outDataset.SetProjection(projection)     
    outBand = outDataset.GetRasterBand(1)
    outBand.WriteArray(cmap,0,0) 
    outBand.FlushCache() 
    print 'last change map written to: %s'%outfn1  
    outfn2=outfn.replace(name,name+'_fmap')
    outDataset = driver.Create(outfn2,cols,rows,1,GDT_Byte)
    if geotransform is not None:
        outDataset.SetGeoTransform(geotransform)      
    if projection is not None:
        outDataset.SetProjection(projection)     
    outBand = outDataset.GetRasterBand(1)
    outBand.WriteArray(fmap,0,0) 
    outBand.FlushCache() 
    print 'frequency map written to: %s'%outfn2     
    outfn3=outfn.replace(name,name+'_bmap')
    outDataset = driver.Create(outfn3,cols,rows,k-1,GDT_Byte)
    if geotransform is not None:
        outDataset.SetGeoTransform(geotransform)       
    if projection is not None:
        outDataset.SetProjection(projection)  
    for i in range(k-1):        
        outBand = outDataset.GetRasterBand(i+1)
        outBand.WriteArray(bmap[:,:,i],0,0) 
        outBand.FlushCache() 
    print 'bitemporal map image written to: %s'%outfn3    
    outfn4=outfn.replace(name,name+'_smap')
    outDataset = driver.Create(outfn4,cols,rows,1,GDT_Byte)
    if geotransform is not None:
        outDataset.SetGeoTransform(geotransform)      
    if projection is not None:
        outDataset.SetProjection(projection)     
    outBand = outDataset.GetRasterBand(1)
    outBand.WriteArray(smap,0,0) 
    outBand.FlushCache() 
    print 'first change map written to: %s'%outfn4         
    print 'total elapsed time: '+str(time.time()-start)   
    outDataset = None    
    inDataset1 = None        
    
if __name__ == '__main__':
    main()     