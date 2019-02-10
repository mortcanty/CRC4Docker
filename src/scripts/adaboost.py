#!/usr/bin/env python
#******************************************************************************
#  Name:     adaboost.py
#  Purpose:  supervised classification of multispectral images with ADABOOST.M1
#  Usage:             
#    python adaboost.py
#
# Copyright (c) 2018 Mort Canty

import auxil.supervisedclass as sc
import auxil.readshp as rs
import gdal, os, time, sys, getopt
from osgeo.gdalconst import GA_ReadOnly, GDT_Byte
import matplotlib.pyplot as plt
import numpy as np

def seq_class(ffns,Xs,alphas,K):
#  weighted classification of observations Xs with list of FFN classifiers
#  returns labels, class membership probabilities   
    M = len(ffns)
    _,ps1 = ffns[0].classify(Xs)
    ps = alphas[0]*ps1
    for i in range(1,M):
        _,ps1 = ffns[i].classify(Xs)
        ps += alphas[i]*ps1
    den = np.sum(ps,0)
    for i in range(K):
        ps[i,:] = ps[i,:]/den
    labels = np.argmax(ps,1)
    return (labels,ps)    

class Ffnekfab(sc.Ffnekf):
    
    def __init__(self,Gs,ls,p,L,epochs=5):
        sc.Ffnekf.__init__(self,Gs,ls,L,epochs)
        tmp = np.roll(np.cumsum(p),1)
        tmp[0] = 0.0
        self._sd = tmp
        
    def train(self):
        try:
    #      update matrices for hidden and output weight  
            dWh = np.zeros((self._N+1,self._L))   
            dWo = np.zeros((self._L+1,self._K))
            cost = []
            costv = []
            itr = 0
            epoch = 0  
            maxitr = self._epochs*self._m
            while itr < maxitr: 
    #          select training pair from distribution d
                nu = np.sum(np.where(self._sd < np.random.rand(),1,0))-1
                x = self._Gs[:,nu]
                y = self._ls[:,nu]
    #          forward pass
                m = self.forwardpass(x)    
    #          output error
                e = y - m  
    #          loop over output neurons
                for k in range(self._K):
    #              linearized input
                    Ao  = m[k,0]*(1-m[k,0])*self._n   
    #              Kalman gain
                    So = self._So[:,:,k]  
                    SA = So*Ao
                    Ko = SA/((Ao.T*SA)[0] + 1)
    #              determine delta for this neuron
                    dWo[:,k] = (Ko*e[k,0]).ravel()
    #              update its covariance matrix
                    So -= Ko*Ao.T*So  
                    self._So[:,:,k] = So  
    #          update the output weights
                self._Wo = self._Wo + dWo                           
    #          backpropagated error
                beta_o = e.A*m.A*(1-m.A) 
    #          loop over hidden neurons
                for j in range(self._L):
    #              linearized input
                    Ah = x*(self._n)[j+1,0]*(1-self._n[j+1,0])
    #              Kalman gain
                    Sh = self._Sh[:,:,j]  
                    SA = Sh*Ah  
                    Kh = SA/((Ah.T*SA)[0] + 1)                        
    #              determine delta for this neuron
                    dWh[:,j] = (Kh*(self._Wo[j+1,:]*beta_o)).ravel()
    #              update its covariance matrix
                    Sh -= Kh*Ah.T*Sh
                    self._Sh[:,:,j] = Sh
    #          update the hidden weights
                self._Wh = self._Wh + dWh  
                if itr % self._m == 0:
                    cost.append(self.cost())
                    costv.append(self.costv())
                    epoch += 1 
                itr += 1
            return (cost,costv)  
        except Exception as e:
            print 'Error: %s'%e 
            return None                        
    
def main():    
    usage = '''
Usage:
------------------------------------------------

supervised classification of multispectral images with ADABOOST.M1

python %s [OPTIONS] filename trainShapefile
    
Options:

   -h         this help
   -p <list>  band positions e.g. -p [1,2,3,4] 
   -L <int>  number of hidden neurons (default 10)
   -n <int>   number of nnet instances (default 50)
   -e <int>   epochs for ekf training (default 3)
   
If the input file is named 

         path/filenbasename.ext then

The output classification file is named 

         path/filebasename_class.ext

------------------------------------------------''' %sys.argv[0]

    outbuffer = 100

    options, args = getopt.getopt(sys.argv[1:],'hp:n:e:L:')
    pos = None  
    L = [10]
    epochs = 3
    instances = 50
    for option, value in options:
        if option == '-h':
            print usage
            return
        elif option == '-p':
            pos = eval(value) 
        elif option == '-e':
            epochs = eval(value)    
        elif option == '-n':
            instances = eval(value)                          
        elif option == '-L':
            L = [eval(value)]                           
    if len(args) != 2: 
        print 'Incorrect number of arguments'
        print usage
        sys.exit(1)      
    print 'Training with ADABOOST.M1 and %i epochs per ffn'%epochs          
    infile = args[0]  
    trnfile = args[1]      
    gdal.AllRegister() 
    if infile:                   
        inDataset = gdal.Open(infile,GA_ReadOnly)
        cols = inDataset.RasterXSize
        rows = inDataset.RasterYSize    
        bands = inDataset.RasterCount
        geotransform = inDataset.GetGeoTransform()
    else:
        return  
    if pos is None: 
        pos = range(1,bands+1)  
    N = len(pos)  
    rasterBands = [] 
    for b in pos:
        rasterBands.append(inDataset.GetRasterBand(b))     
#  output file
    path = os.path.dirname(infile)
    basename = os.path.basename(infile)
    root, ext = os.path.splitext(basename)
    outfile = '%s/%s_class%s'%(path,root,ext)   
#  setup output class image dataset 
    driver = inDataset.GetDriver() 
    outDataset = driver.Create(outfile,cols,rows,1,GDT_Byte) 
    projection = inDataset.GetProjection()
    if geotransform is not None:
        outDataset.SetGeoTransform(geotransform)
    if projection is not None:
        outDataset.SetProjection(projection)    
    outBand = outDataset.GetRasterBand(1)                    
#  get the training data        
    Xs,Ls,K,_ = rs.readshp(trnfile,inDataset,pos) 
    m = Ls.shape[0]  
#  stretch the pixel vectors to [-1,1] 
    maxx = np.max(Xs,0)
    minx = np.min(Xs,0)
    for j in range(len(pos)):
        Xs[:,j] = 2*(Xs[:,j]-minx[j])/(maxx[j]-minx[j]) - 1.0 
#  random permutation of training data
    idx = np.random.permutation(m)
    Xs = Xs[idx,:] 
    Ls = Ls[idx,:]     
#  train on 2/3 of training examples, rest for testing
    mtrn =  int(0.67*m)  
    mtst = m-mtrn 
    Xstrn = Xs[:mtrn,:]
    Lstrn = Ls[:mtrn,:] 
    Xstst = Xs[mtrn:,:]  
    Lstst = Ls[mtrn:,:] 
    labels_train = np.argmax(Lstrn,1)
    labels_test = np.argmax(Lstst,1)
#  list of network instances, weights and errors   
    ffns = []
    alphas = []
    errtrn = []
    errtst = []
#  initial probability distribution
    p = np.ones(mtrn)/mtrn    
#  loop through the network instance
    start = time.time()
    instance = 1
    while instance<instances:
        trial = 1
        while trial < 6:
            print 'running instance: %i  trial: %i' \
                                  %(instance,trial)
#          instantiate a ffn and train it  
            ffn = Ffnekfab(Xstrn,Lstrn,p,L,epochs)         
            ffn.train()
#          determine beta            
            labels,_ = ffn.classify(Xstrn)
            labels -= 1
            idxi = np.where(labels != labels_train)[0]
            idxc = np.where(labels == labels_train)[0]
            epsilon = np.sum(p[idxi])
            beta = epsilon/(1-epsilon)
            if beta < 1.0:
#              continue                   
                ffns.append(ffn)
                alphas.append(np.log(1.0/beta))
#              update distribution
                p[idxc] = p[idxc]*beta  
                p = p/np.sum(p)     
#              train error                
                labels,_=seq_class(ffns,Xstrn,alphas,K)
                tmp=np.where(labels!=labels_train,1,0)     
                errtrn.append(np.sum(tmp)/float(mtrn))       
#              test error                
                labels,_=seq_class(ffns,Xstst,alphas,K)
                tmp = np.where(labels!=labels_test,1,0)     
                errtst.append(np.sum(tmp)/float(mtst))      
                print 'train error: %f test error: %f'\
                     %(errtrn[-1],errtst[-1])
#              this instance is done                
                trial = 6
                instance += 1            
            else:
                trial += 1
#              break off training               
                if trial==6:
                    instance = instances      
    print 'elapsed time %s' %str(time.time()-start)
#  plot errors
    n = len(errtrn)
    errtrn = np.array(errtrn)
    errtst = np.array(errtst)
    x = np.arange(1,n+1,1)
    ax = plt.subplot(111)
    ax.semilogx(x,errtrn,label='train')
    ax.semilogx(x,errtst,label='test') 
    ax.legend()
    ax.set_xlabel('number of networks')
    ax.set_ylabel('classification error')
    plt.savefig('/home/mort/LaTeX/new projects/CRC4/Chapter7/fig7_3.eps',bbox_inches='tight')
    plt.show()
#  classify the image           
    print 'classifying...'
    start = time.time()
    tile = np.zeros((outbuffer*cols,N),dtype=np.float32)    
    for row in range(rows/outbuffer):
        print 'row: %i'%(row*outbuffer)
        for j in range(N):
            tile[:,j] = rasterBands[j].ReadAsArray(0,row*outbuffer,cols,outbuffer).ravel()
            tile[:,j] = 2*(tile[:,j]-minx[j])/(maxx[j]-minx[j]) - 1.0               
        cls, _ = seq_class(ffns,tile,alphas,K)  
        outBand.WriteArray(np.reshape(cls,(outbuffer,cols)),0,row*outbuffer)
    outBand.FlushCache()    
    print 'thematic map written to: %s'%outfile
    print 'elapsed time %s' %str(time.time()-start)
    
if __name__ == '__main__':
    main()