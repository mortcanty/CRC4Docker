#!/usr/bin/env python
#******************************************************************************
#  Name:     auxil.py
#  Purpose:  auxiliary functions for processing multispectral imagery 
#  Usage:             
#    import auxil

import numpy as np  
import math, ctypes  
from numpy.ctypeslib import ndpointer
from scipy.special import betainc 
from numpy.fft import fft2, ifft2, fftshift 
import scipy.ndimage.interpolation as ndii 

# wrap the provisional means dll
lib = ctypes.cdll.LoadLibrary('libprov_means.so')    
provmeans = lib.provmeans 
provmeans.restype = None   
c_double_p = ctypes.POINTER(ctypes.c_double) 
provmeans.argtypes = [ndpointer(np.float64), 
                      ndpointer(np.float64),
                      ctypes.c_int,
                      ctypes.c_int,
                      c_double_p,
                      ndpointer(np.float64),
                      ndpointer(np.float64)]  

# color table
ctable = [ 0,0,0,       255,0,0,    0,255,0,     0,0,255, \
           255,255,0,   0,255,255,  255,0,255,   176,48,96, \
           46,139,87,   255,0,0,    205,0,0,     0,255,0, \
           0,200,200,   85,107,47,  189,183,107, 255,160,122, \
           240,128,128, 255,69,0]

# ---------------------
# orthogonal regression
# ---------------------

def orthoregress(x,y): 
    Xm = np.mean(x)
    Ym = np.mean(y) 
    s = np.cov(x,y)
    R = s[0,1]/math.sqrt(s[1,1]*s[0,0])
    lam,vs = np.linalg.eig(s)         
    idx = np.argsort(lam)    
    vs = vs[:,idx]      # increasing order, so
    b = vs[1,1]/vs[0,1] # first pc is second column
    return [b,Ym-b*Xm,R]

# -------------------------
# F-test for equal variance
# -------------------------

def fv_test(x0,x1):
# taken from IDL library    
    nx0 = len(x0)
    nx1 = len(x1)
    v0 = np.var(x0)
    v1 = np.var(x1)
    if v0 >v1:
        f = v0/v1
        df0 = nx1-1
        df1 = nx0-1
    else:
        f = v1/v0
        df0 = nx1-1
        df1 = nx0-1
    prob = 2.0*betainc(0.5*df1,0.5*df0,df1/(df1+df0*f))
    if prob >1:
        return (f,2.0-prob)
    else:
        return (f,prob) 

# ------------
# Gauss filter
# ------------

def dist(n,m):
    ''''n cols x m rows distance array'''    
    result = []
    for i in range(m):
        for j in range(n):
            y = float(min(i,m-i))
            x = float(min(j,n-j))
            result.append(math.sqrt(x**2+y**2))
    return result

def gaussfilter(sigma,n,m):
    dst = dist(n,m)
    result = []
    for d in dst:
        result.append( math.exp(-d**2/(2*sigma**2)) )
    return np.reshape(np.array(result),(m,n)) 

# -----------------
# provisional means
# -----------------

class Cpm(object):
    '''Provisional means algorithm'''
    def __init__(self,N):
        self.mn = np.zeros(N)
        self.cov = np.zeros((N,N))
        self.sw = 0.0000001
         
    def update(self,Xs,Ws=None):
        lngth = len(np.shape(Xs))
        if lngth==2:
            n,N = np.shape(Xs)  
        else:
            N = len(Xs)  
            n = 1   
        if Ws is None:
            Ws = np.ones(n)
        sw = ctypes.c_double(self.sw)        
        mn = self.mn
        cov = self.cov
        provmeans(Xs,Ws,N,n,ctypes.byref(sw),mn,cov)
        self.sw = sw.value
        self.mn = mn
        self.cov = cov
          
    def covariance(self):
        c = np.mat(self.cov/(self.sw-1.0))
        d = np.diag(np.diag(c))
        return c + c.T - d
    
    def means(self):
        return self.mn                     

# ---------
# kernels
# ---------

def kernelMatrix(X,Y=None,gma=None,nscale=10,kernel=0):
    if Y is None:
        Y = X
    if kernel == 0:
        X = np.mat(X)
        Y = np.mat(Y)    
        return (X*(Y.T),0)
    else:
        m = X[:,0].size
        n = Y[:,0].size
        onesm = np.mat(np.ones(m))
        onesn = np.mat(np.ones(n))
        K = np.mat(np.sum(X*X,axis=1)).T*onesn
        K = K + onesm.T*np.mat(np.sum(Y*Y,axis=1))
        K = K - 2*np.mat(X)*np.mat(Y).T
        if gma is None:
            scale = np.sum(np.sqrt(abs(K)))/(m**2-m) 
            gma = 1/(2*(nscale*scale)**2)   
        return (np.exp(-gma*K),gma)
    
def center(K):
    m = K[:,0].size
    Imm = np.mat(np.ones((m,m)))
    return K - (Imm*K + K*Imm - np.sum(K)/m)/m      

# ------------------------    
# generalized eigenproblem
# ------------------------   
def choldc(A):
    '''Cholesky-Banachiewicz algorithm, 
       A is a numpy matrix'''
    L = A - A  
    for i in range(len(L)):
        for j in range(i):
            sm = 0.0
            for k in range(j):
                sm += L[i,k]*L[j,k]
            L[i,j] = (A[i,j]-sm)/L[j,j]
        sm = 0.0
        for k in range(i):
            sm += L[i,k]*L[i,k]
        L[i,i] = math.sqrt(A[i,i]-sm)        
    return L               
        
def geneiv(A,B): 
    '''solves A*x = lambda*B*x for numpy matrices A, B 
       returns eigenvectors in columns'''
    Li = np.linalg.inv(choldc(B))
    C = Li*A*(Li.transpose())
    C = np.asmatrix((C + C.transpose())*0.5,np.float32)
    lambdas,V = np.linalg.eig(C)
    return lambdas, Li.transpose()*V     

def similarity(bn0, bn1):
    """Register bn1 to bn0 ,  M. Canty 2012
bn0, bn1 and returned result are image bands      
Modified from Imreg.py, see http://www.lfd.uci.edu/~gohlke/:
 Copyright (c) 2011-2012, Christoph Gohlke
 Copyright (c) 2011-2012, The Regents of the University of California
 Produced at the Laboratory for Fluorescence Dynamics
 All rights reserved.    
    """
 
    def highpass(shape):
        """Return highpass filter to be multiplied with fourier transform."""
        x = np.outer(
                        np.cos(np.linspace(-math.pi/2., math.pi/2., shape[0])),
                        np.cos(np.linspace(-math.pi/2., math.pi/2., shape[1])))
        return (1.0 - x) * (2.0 - x)    

    def logpolar(image, angles=None, radii=None):
        """Return log-polar transformed image and log base."""
        shape = image.shape
        center = shape[0] / 2, shape[1] / 2
        if angles is None:
            angles = shape[0]
            if radii is None:
                radii = shape[1]
        theta = np.empty((angles, radii), dtype=np.float64)
        theta.T[:] = -np.linspace(0, np.pi, angles, endpoint=False)
#      d = radii
        d = np.hypot(shape[0]-center[0], shape[1]-center[1])
        log_base = 10.0 ** (math.log10(d) / (radii))
        radius = np.empty_like(theta)
        radius[:] = np.power(log_base, np.arange(radii,
                                                   dtype=np.float64)) - 1.0
        x = radius * np.sin(theta) + center[0]
        y = radius * np.cos(theta) + center[1]
        output = np.empty_like(x)
        ndii.map_coordinates(image, [x, y], output=output)
        return output, log_base
             
    lines0,samples0 = bn0.shape
#  make reference and warp bands same shape    
    bn1 = bn1[0:lines0,0:samples0]   
#  get scale, angle      
    f0 = fftshift(abs(fft2(bn0)))
    f1 = fftshift(abs(fft2(bn1)))
    h = highpass(f0.shape)
    f0 *= h
    f1 *= h
    del h
    f0, log_base = logpolar(f0)
    f1, log_base = logpolar(f1)
    f0 = fft2(f0)
    f1 = fft2(f1)
    r0 = abs(f0) * abs(f1)
    ir = abs(ifft2((f0 * f1.conjugate()) / r0))
    i0, i1 = np.unravel_index(np.argmax(ir), ir.shape)
    angle = 180.0 * i0 / ir.shape[0]
    scale = log_base ** i1 
    if scale > 1.8:
        ir = abs(ifft2((f1 * f0.conjugate()) / r0))
        i0, i1 = np.unravel_index(np.argmax(ir), ir.shape)
        angle = -180.0 * i0 / ir.shape[0]
        scale = 1.0 / (log_base ** i1)
        if scale > 1.8:
            raise ValueError("Images are not compatible. Scale change > 1.8")
    if angle < -90.0:
        angle += 180.0
    elif angle > 90.0:
        angle -= 180.0           
#  re-scale and rotate and then get shift                   
    bn2 = ndii.zoom(bn1, 1.0/scale)
    bn2 = ndii.rotate(bn2, angle)
    if bn2.shape < bn0.shape:
        t = np.zeros_like(bn0)
        t[:bn2.shape[0], :bn2.shape[1]] = bn2
        bn2 = t
    elif bn2.shape > bn0.shape:
        bn2 = bn2[:bn0.shape[0], :bn0.shape[1]] 
    f0 = fft2(bn0)
    f1 = fft2(bn2)
    ir = abs(ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1))))
    t0, t1 = np.unravel_index(np.argmax(ir), ir.shape)
    if t0 > f0.shape[0] // 2:
        t0 -= f0.shape[0]
    if t1 > f0.shape[1] // 2:
        t1 -= f0.shape[1]                                               
#  return result   
    return (scale,angle,[t0,t1])                 

# ---------------------------
# discrete wavelet transform
# ---------------------------

class DWTArray(object):
    '''Partial DWT representation of image band
       which is input as 2-D uint8 array'''
    def __init__(self,band,samples,lines,itr=0):
# Daubechies D4 wavelet       
        self.H = np.asarray([(1-math.sqrt(3))/8,(3-math.sqrt(3))/8,(3+math.sqrt(3))/8,(1+math.sqrt(3))/8])   
        self.G = np.asarray([-(1+math.sqrt(3))/8,(3+math.sqrt(3))/8,-(3-math.sqrt(3))/8,(1-math.sqrt(3))/8])
        self.num_iter = itr        
        self.max_iter = 3
# ignore edges if band dimension is not divisible by 2^max_iter        
        r = 2**self.max_iter
        self.samples = r*(samples//r)
        self.lines = r*(lines//r)  
        self.data = np.asarray(band[:self.lines,:self.samples],np.float32) 
        
    def get_quadrant(self,quadrant,float=False):
        if self.num_iter==0:
            m = 2*self.lines
            n = 2*self.samples         
        else:
            m = self.lines/2**(self.num_iter-1)
            n = self.samples/2**(self.num_iter-1)
        if quadrant == 0:
            f = self.data[:m/2,:n/2]
        elif quadrant == 1:
            f = self.data[:m/2,n/2:n]
        elif quadrant == 2:
            f = self.data[m/2:m,:n/2]
        else:
            f = self.data[m/2:m,n/2:n]
        if float: 
            return f
        else:   
            f = np.where(f<0,0,f) 
            f = np.where(f>255,255,f)   
            return np.asarray(f,np.uint8)    
    
    def put_quadrant(self,f1,quadrant):
        if not (quadrant in range(4)) or (self.num_iter==0):
            return 0      
        m = self.lines/2**(self.num_iter-1)
        n = self.samples/2**(self.num_iter-1)
        f0 = self.data
        if quadrant == 0:
            f0[:m/2,:n/2] = f1
        elif quadrant == 1:
            f0[:m/2,n/2:n] = f1      
        elif quadrant == 2:
            f0[m/2:m,:n/2] = f1
        else:
            f0[m/2:m,n/2:m] = f1             
        return 1     
          
    def normalize(self,a,b): 
#      normalize wavelet coefficients at all levels        
        for c in range(1,self.num_iter+1):
            m = self.lines/(2**c)
            n = self.samples/(2**c) 
            self.data[:m,n:2*n]    = a[0]*self.data[:m,n:2*n]+b[0]                            
            self.data[m:2*m,:n]    = a[1]*self.data[m:2*n,:n]+b[1]
            self.data[m:2*m,n:2*n] = a[2]*self.data[m:2*n,n:2*n]+b[2]
            
    def filter(self):
#      single application of filter bank          
        if self.num_iter == self.max_iter:
            return 0
#      get upper left quadrant       
        m = self.lines/2**self.num_iter
        n = self.samples/2**self.num_iter       
        f0 = self.data[:m,:n]  
#      temporary arrays
        f1 = np.zeros((m/2,n)) 
        g1 = np.zeros((m/2,n))
        ff1 = np.zeros((m/2,n/2))
        fg1 = np.zeros((m/2,n/2))
        gf1 = np.zeros((m/2,n/2))
        gg1 = np.zeros((m/2,n/2))
#      filter columns and downsample        
        ds = np.asarray(range(m/2))*2+1
        for i in range(n):
            temp = np.convolve(f0[:,i].ravel(),\
                                       self.H,'same')
            f1[:,i] = temp[ds]
            temp = np.convolve(f0[:,i].ravel(),\
                                       self.G,'same')
            g1[:,i] = temp[ds]               
#      filter rows and downsample
        ds = np.asarray(range(n/2))*2+1
        for i in range(m/2):
            temp = np.convolve(f1[i,:],self.H,'same')
            ff1[i,:] = temp[ds]
            temp = np.convolve(f1[i,:],self.G,'same')
            fg1[i,:] = temp[ds]  
            temp = np.convolve(g1[i,:],self.H,'same')
            gf1[i,:] = temp[ds]                      
            temp = np.convolve(g1[i,:],self.G,'same')
            gg1[i,:] = temp[ds]        
        f0[:m/2,:n/2] = ff1
        f0[:m/2,n/2:] = fg1
        f0[m/2:,:n/2] = gf1
        f0[m/2:,n/2:] = gg1
        self.data[:m,:n] = f0       
        self.num_iter = self.num_iter+1        
            
    def invert(self):
        H = self.H[::-1]   
        G = self.G[::-1]
        m = self.lines/2**(self.num_iter-1)
        n = self.samples/2**(self.num_iter-1)
#      get upper left quadrant      
        f0 = self.data[:m,:n]     
        ff1 = f0[:m/2,:n/2]
        fg1 = f0[:m/2,n/2:] 
        gf1 = f0[m/2:,:n/2]
        gg1 = f0[m/2:,n/2:]
        f1 = np.zeros((m/2,n))
        g1 = np.zeros((m/2,n))
#      upsample and filter rows
        for i in range(m/2):            
            a = np.ravel(np.transpose(np.vstack((ff1[i,:],np.zeros(n/2)))))
            b = np.ravel(np.transpose(np.vstack((fg1[i,:],np.zeros(n/2)))))
            f1[i,:] = np.convolve(a,H,'same') + np.convolve(b,G,'same')
            a = np.ravel(np.transpose(np.vstack((gf1[i,:],np.zeros(n/2)))))
            b = np.ravel(np.transpose(np.vstack((gg1[i,:],np.zeros(n/2)))))            
            g1[i,:] = np.convolve(a,H,'same') + np.convolve(b,G,'same')        
#      upsample and filter columns
        for i in range(n):
            a = np.ravel(np.transpose(np.vstack((f1[:,i],np.zeros(m/2)))))
            b = np.ravel(np.transpose(np.vstack((g1[:,i],np.zeros(m/2)))))          
            f0[:,i] = 4*(np.convolve(a,H,'same') + np.convolve(b,G,'same'))
        self.data[:m,:n] = f0                                   
        self.num_iter = self.num_iter-1    
         
        
class ATWTArray(object):
    '''A trous wavelet transform'''
    def __init__(self,band):
        self.num_iter = 0
#      cubic spline filter       
        self.H = np.array([1.0/16,1.0/4,3.0/8,1.0/4,1.0/16])
#      data arrays
        self.lines,self.samples = band.shape
        self.bands = np.zeros((4,self.lines,self.samples),np.float32)
        self.bands[0,:,:] = np.asarray(band,np.float32)
    
    def inject(self,band):
        m = self.lines
        n = self.samples
        self.bands[0,:,:] = band[0:m,0:n]
        
    def get_band(self,i):
        return self.bands[i,:,:]
        
    def normalize(self,a,b):
        if self.num_iter > 0:
            for i in range(1,self.num_iter+1):
                self.bands[i,:,:] = a*self.bands[i,:,:]+b
                
    def filter(self):
        if self.num_iter < 3:
            self.num_iter += 1     
#          a trous filter       
            n = 2**(self.num_iter-1)
            H = np.vstack((self.H,np.zeros((2**(n-1),5))))
            H = np.transpose(H).ravel()
            H = H[0:-n]
#          temporary arrays
            f1 = np.zeros((self.lines,self.samples))
            ff1 = f1*0.0
#          filter columns
            f0 = self.bands[0,:,:]
#          filter columns
            for i in range(self.samples):
                f1[:,i] = np.convolve(f0[:,i].ravel(), H, 'same')
#          filter rows
            for j in range(self.lines):
                ff1[j,:] = np.convolve(f1[j,:], H, 'same')
            self.bands[self.num_iter,:,:] = self.bands[0,:,:] - ff1
            self.bands[0,:,:] = ff1
            
    def invert(self):
        if self.num_iter > 0:
            self.bands[0,:,:] += self.bands[self.num_iter,:,:]
            self.num_iter -= 1  
            
# --------------------
# contrast enhancement
# -------------------
def linstr(x):
# linear stretch
    return bytestr(x)
    
def histeqstr(x):
    x = bytestr(x)
#  histogram equalization stretch
    hist,bin_edges = np.histogram(x,256,(0,256))
    cdf = hist.cumsum()
    lut = 255*cdf/float(cdf[-1])
    return np.interp(x,bin_edges[:-1],lut)

def lin2pcstr(x):
#  2% linear stretch
    x = bytestr(x)
    hist,bin_edges = np.histogram(x,256,(0,256))
    cdf = hist.cumsum()
    lower = 0
    i = 0
    while cdf[i] < 0.02*cdf[-1]:
        lower += 1
        i += 1
    upper = 255
    i = 255
    while cdf[i] > 0.98*cdf[-1]:
        upper -= 1
        i -= 1
    fp = (bin_edges-lower)*255/(upper-lower)
    fp = np.where(bin_edges<=lower,0,fp)
    fp = np.where(bin_edges>=upper,255,fp)
    return np.interp(x,bin_edges,fp)     

def bytestr(arr,rng=None):
#  byte stretch image numpy array
    shp = arr.shape
    arr = arr.ravel()
    if rng is None:
        rng = [np.min(arr),np.max(arr)]
    tmp =  (arr-rng[0])*255.0/(rng[1]-rng[0])
    tmp = np.where(tmp<0,0,tmp)  
    tmp = np.where(tmp>255,255,tmp) 
    return np.asarray(np.reshape(tmp,shp),np.uint8)  

def rebin(a, new_shape):
    M, N = a.shape
    m, n = new_shape
    if m<M:
        return a.reshape((m,M/m,n,N/n)).mean(3).mean(1)
    else:
        return np.repeat(np.repeat(a, m/M, axis=0), n/N, axis=1)
            
if __name__ == '__main__':
    pass
                
        