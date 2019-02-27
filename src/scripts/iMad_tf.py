#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly, GDT_Float32
import getopt, sys, os, time

tfd = tfp.distributions

def read_image(fn,dims=None,pos=None):
#  read image into data matrix    
    gdal.AllRegister()
    inDataset = gdal.Open(fn,GA_ReadOnly)
    cols = inDataset.RasterXSize
    rows = inDataset.RasterYSize    
    bands = inDataset.RasterCount
    if dims:
        x0,y0,cols,rows = tuple(dims)
    else:
        x0 = 0
        y0 = 0
    if pos is not None:
        bands = len(pos)
    else:
        pos = range(1,bands+1)           
    G = np.zeros((rows*cols,bands))
    k = 0                               
    for b in pos:
        band = inDataset.GetRasterBand(b)
        tmp = band.ReadAsArray(x0,y0,cols,rows)\
                              .astype(float).ravel()
        G[:,k] = tmp - np.mean(tmp)
        k += 1      
    return (G,cols,rows,bands,inDataset,x0,y0)

def tf_covw(x,ws):
    '''weighted covariance matrix and weighted means of uncentered data tensor x'''  
    x = tf.transpose(x) # transposed data matrix
    S = tf.shape(x)
    N = S[0]
    sumw = tf.reduce_sum(ws)
    ws = tf.reshape( tf.tile(ws,[N]), S ) 
    xw = tf.multiply(x,ws)
    mx = tf.divide( tf.reduce_sum(xw, axis=1, keepdims=True), sumw )
    mx1 = tf.matmul(mx,tf.transpose(mx))
    xw = tf.multiply(x,tf.sqrt(ws))
    vx = tf.matmul(xw, tf.transpose(xw))/sumw
    return vx-mx1, mx

def geneiv(A,B):
    '''solves A*x = lambda*B*x for tensors A, B 
       returns eigenvectors in columns'''
    Li = tf.linalg.inv(tf.cholesky(B))
    C = tf.matmul(tf.matmul(Li,A),Li,transpose_b=True)
    lambdas,V = tf.linalg.eigh(C)
    return lambdas, tf.matmul(tf.transpose(Li),V)

def imad(x1,x2,pvs,niter):  
    ''' IR.MAD algorithm'''
    m = tf.shape(x1)[0]
    N = tf.shape(x1)[1]
    x = tf.concat([x1,x2],axis=1)
    itr = 0
    while itr<niter:
        itr += 1 
    #  weighted covariance and means    
        cov,ms = tf_covw(x,pvs)
        ms1 = tf.transpose(ms[:N]) #row vectors
        ms2 = tf.transpose(ms[N:])
        s11 = cov[:N,:N]
        s12 = cov[:N,N:]
        s21 = cov[N:,:N]
        s22 = cov[N:,N:]
        c1 = tf.matmul(tf.matmul(s12,tf.linalg.inv(s22)),s21)
        b1 = s11
        c2 = tf.matmul(tf.matmul(s21,tf.linalg.inv(s11)),s12)
        b2 = s22
        rho2,A = geneiv(c1,b1)
        _   ,B = geneiv(c2,b2)
        rho = tf.sqrt(rho2[::-1])
        A = A[:,::-1]  
        B = B[:,::-1]
    #  ensure positive correlation between each pair of canonical variates        
        cov = tf.diag_part(tf.matmul(tf.matmul(tf.transpose(A),s12),B))
        cov = tf.diag(tf.divide(cov,tf.abs(cov)))
        B = tf.matmul(B,cov)  
    #  calculate p-value weights
        sig2s = 2*(1-rho)
        sig2s = tf.reshape( tf.tile(sig2s,[m]), (m,N) )
        ms1 = tf.reshape( tf.tile(ms1[0],[m]), (m,N) )
        ms2 = tf.reshape( tf.tile(ms2[0],[m]), (m,N) )
        CV1 = tf.matmul( x1-ms1, A )
        CV2 = tf.matmul( x2-ms2, B )
        MADs = CV1 - CV2
        chisqr = tf.reduce_sum(tf.square(MADs)/sig2s, axis=1) 
        N1 = tf.cast(N,dtype=tf.float64)
        one = tf.constant(1.0,dtype=tf.float64)
        pvs = tf.subtract(one,tfd.Chi2(N1).cdf(chisqr))
    return (MADs, chisqr, rho)

def main():   
    usage = '''
Usage:
------------------------------------------------
Run the iterated MAD algorithm on two multispectral images
on the tensorflow API   

python %s [OPTIONS] filename1 filename2
    
Options:
   -h           this help
   -i  <int>    maximum iterations (default 50)
   -d  <list>   spatial subset list e.g. -d [0,0,500,500]
   -p  <list>   spectral subset list e.g. -p [1,2,3,4] 
   -s  <string> TF session e.g. -s grpc://localhost:2222 (defaults to default_session)
   -D  <string> pi operations to a device e.g. -D /job:worker/task:0/gpu:0 
-----------------------------------------------------''' %sys.argv[0]
    options, args = getopt.getopt(sys.argv[1:],'hi:d:p:s:D:')
    dims = None 
    pos = None 
    niter = 50
    session = tf.get_default_session()
    dev = None
    for option, value in options:
        if option == '-h':
            print usage
            return
        elif option == '-i':
            niter = eval(value) 
        elif option == '-d':
            dims = eval(value) 
        elif option == '-p':
            pos = eval(value)
        elif option == '-s':
            session = str(value)
        elif option == '-D':
            dev = str(value)
            
    if len(args) != 2:
        print 'Incorrect number of arguments'
        print usage
        return                                    
    fn1 = args[0]
    fn2 = args[1]
    path = os.path.dirname(fn1)
    basename1 = os.path.basename(fn1)
    root1, _ = os.path.splitext(basename1)
    basename2 = os.path.basename(fn2)
    root2, _ = os.path.splitext(basename2)
    outfn = path + '/' + 'MAD(%s-%s)%s'%(root1,root2,'.tif') 
    
    print '------------IRMAD (tensorflow) -------------'
    print time.asctime()     
    print 'first scene:  '+fn1
    print 'second scene: '+fn2   
    start = time.time()    

#  the graph
    with tf.device(dev):    
        x1 = tf.placeholder(tf.float64)
        x2 = tf.placeholder(tf.float64)
        ws = tf.placeholder(tf.float64)
        
        imad_op = imad(x1,x2,ws,niter)    

    img1,_,_,_,_,_,_ = read_image(fn1,dims=dims,pos=pos)
    img2,cols,rows,bands,inDataset,x0,y0 = read_image(fn2,dims=dims,pos=pos)
    m,_ = img1.shape
    pvs = np.ones(m)
    
    with tf.Session(session) as sess:  
        sess.run(tf.global_variables_initializer())  
        MADs,chisqr,rho = sess.run(imad_op,feed_dict = {x1:img1,x2:img2,ws:pvs})
    MADs = np.reshape(MADs,(rows,cols,bands))
    chisqr = np.reshape(chisqr,(rows,cols))
    
    print 'canonical corr: %s'%str(rho)
    
    driver = gdal.GetDriverByName('GTiff')
    outDataset = driver.Create(outfn,
                cols,rows,bands+1,GDT_Float32)      
    
    projection = inDataset.GetProjection()
    geotransform = inDataset.GetGeoTransform()
    if geotransform is not None:
        gt = list(geotransform)
        gt[0] = gt[0] + x0*gt[1]
        gt[3] = gt[3] + y0*gt[5]
        outDataset.SetGeoTransform(tuple(gt))
    if projection is not None:
        outDataset.SetProjection(projection)     
        
    for k in range(bands):        
        outBand = outDataset.GetRasterBand(k+1)
        outBand.WriteArray(MADs[:,:,k],0,0) 
        outBand.FlushCache() 
    outBand = outDataset.GetRasterBand(bands+1)
    outBand.WriteArray(chisqr,0,0) 
    outBand.FlushCache()
    print 'MAD variates written to: %s'%outfn   
    print 'elapsed time: %s'%str(time.time()-start)
    outDataset = None
    
if __name__ == '__main__':
    main()    

