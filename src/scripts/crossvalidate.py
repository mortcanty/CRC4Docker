#!/usr/bin/env python
#******************************************************************************
#  Name:     crossvalidate.py
#  Purpose:  IPython parallelized cross-validation  
#  Usage:             
#    python crossvalidate.py [options] infile trainshapefile
# 
#  Copyright (c) 2018, Mort Canty

import auxil.readshp as rs
from ipyparallel import Client
import gdal, time, sys, getopt
from osgeo.gdalconst import GA_ReadOnly
import numpy as np

def crossvalidate((Gstrn,lstrn,Gstst,lstst,L,trainalg)):
    import auxil.supervisedclass as sc
    if   trainalg == 1:
        classifier = sc.Maxlike(Gstrn,lstrn)
    elif trainalg == 2:
        classifier = sc.Gausskernel(Gstrn,lstrn)
    elif trainalg == 3:
        classifier = sc.Ffnbp(Gstrn,lstrn,L)
    elif trainalg == 4:
        classifier = sc.Ffncg(Gstrn,lstrn,L)
    elif trainalg == 5:
        classifier = sc.Dnn(Gstrn,lstrn,L)
    elif trainalg == 6:
        classifier = sc.Svm(Gstrn,lstrn)       
    if classifier.train() is not None:
        return classifier.test(Gstst,lstst)
    else:
        return None
 
def main():    
    usage = '''
Usage: 
---------------------------------------------------------
python %s  [-a algorithm] [-p bands] [-L hidden neurons] infile trainShapefile

bandPositions is a list, e.g., -p [1,2,4]  

algorithm  1=MaxLike
           2=Gausskernel
           3=NNet(backprop)
           4=NNet(congrad)
           5=Dnn(tensorflow)
           6=SVM

prints misclassification rate and standard deviation
--------------------------------------------------------''' %sys.argv[0]
    options, args = getopt.getopt(sys.argv[1:],'hp:a:L:')
    pos = None
    L = [10]
    trainalg = 1
    for option, value in options:
        if option == '-h':
            print usage
            return
        elif option == '-p':
            pos = eval(value)       
        elif option == '-a':
            trainalg = eval(value)
        elif option == '-L':
            L = eval(value)                                 
    if len(args) != 2: 
        print 'Incorrect number of arguments'
        print usage
        sys.exit(1)      
    infile = args[0]  
    trnfile = args[1]      
    gdal.AllRegister()                
    inDataset = gdal.Open(infile,GA_ReadOnly)  
    bands = inDataset.RasterCount
    if pos is None: 
        pos = range(1,bands+1)
    N = len(pos)    
    if trainalg == 1:
        algorithm = 'MaxLike'
    elif trainalg == 2:
        algorithm = 'Gausskernel'    
    elif trainalg == 3:
        algorithm = 'NNet(Backprop)'
    elif trainalg == 4:
        algorithm =  'NNet(Congrad)'
    elif trainalg == 5:
        algorithm =  'Dnn(tensorflow)'    
    else:
        algorithm = 'SVM' 
    print 'Algorithm: %s'%algorithm               
#  get the training data        
    Gs,ls,K,classnames = rs.readshp(trnfile,inDataset,pos) 
    m = ls.shape[0]      
    print str(m) + ' training pixel vectors were read in' 
    
#  stretch the pixel vectors to [-1,1] (for ffn)
    maxx = np.max(Gs,0)
    minx = np.min(Gs,0)
    for j in range(N):
        Gs[:,j] = 2*(Gs[:,j]-minx[j])/(maxx[j]-minx[j]) - 1.0   
#  random permutation of training data
    idx = np.random.permutation(m)
    Gs = Gs[idx,:] 
    ls = ls[idx,:]             

#  cross-validation
    start = time.time()
    rc = Client()   
    print 'submitting cross-validation to %i IPython engines'%len(rc)  
    traintest = []
    for i in range(10):
        sl = slice(i*m//10,(i+1)*m//10)
        traintest.append( (np.delete(Gs,sl,0),np.delete(ls,sl,0), \
                                     Gs[sl,:],ls[sl,:],L,trainalg) )
    v = rc[:]   
    result = v.map(crossvalidate,traintest).get()   
    print 'parallel execution time: %s' %str(time.time()-start)      
    print 'misclassification rate: %f' %np.mean(result)
    print 'standard deviation:     %f' %np.std(result)         
   
if __name__ == '__main__':
    main()