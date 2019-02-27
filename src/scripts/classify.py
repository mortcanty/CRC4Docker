#!/usr/bin/env python
#******************************************************************************
#  Name:     classify.py
#  Purpose:  supervised classification of multispectral images
#  Usage:             
#    python classify.py
#
# Copyright (c) 2018 Mort Canty

import auxil.supervisedclass as sc
import auxil.readshp as rs
import  os, time, sys, getopt
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly, GDT_Byte
import matplotlib.pyplot as plt
import numpy as np

def main():    
    usage = '''
Usage: 
--------------------------------------

Supervised classification of multispectral images

python %s [OPTIONS] filename shapefile 

Options:
  -h            this help
  -p  <list>    RGB band positions to be included
                (default all) e.g. -p [1,2,3]
  -a  <int>     algorithm  1=MaxLike
                           2=Gausskernel
                           3=NNet(backprop)
                           4=NNet(congrad)
                           5=NNet(Kalman)
                           6=Dnn(tensorflow)
                           7=SVM
  -e  <int>     number of epochs (default 100)
  -t  <float>   fraction for training (default 0.67)
  -v            use validation (reserve half of training
                   data for validation)  
  -P            generate class probability image (not
                         available for MaxLike)
  -n            suppress graphical output
  -L  <list>    list of hidden neurons in each 
                   hidden layer (default [10]) 
                            
If the input file is named 

         path/filenbasename.ext then

The output classification file is named 

         path/filebasename_class.ext

the class probabilities output file is named

         path/filebasename_classprobs.ext
         
and the test results file is named

         path/filebasename_<classifier>.tst                            
  
  -------------------------------------'''%sys.argv[0]


    outbuffer = 100

    options, args = getopt.getopt(sys.argv[1:],'hnvPp:t:e:a:L:')
    pos = None
    probs = False   
    L = [10]
    trainalg = 1
    epochs = 100
    graphics = True
    validation = False
    trainfrac = 0.67
    for option, value in options:
        if option == '-h':
            print usage
            return
        elif option == '-p':
            pos = eval(value)
        elif option == '-n':
            graphics = False 
        elif option == '-v':
            validation = True   
        elif option == '-t':
            trainfrac = eval(value)  
        elif option == '-e':
            epochs = eval(value)                          
        elif option == '-a':
            trainalg = eval(value)
        elif option == '-L':
            L = eval(value)    
        elif option == '-P':
            probs = True                              
    if len(args) != 2: 
        print 'Incorrect number of arguments'
        print usage
        sys.exit(1)      
    if trainalg == 1:
        algorithm = 'MaxLike'
    elif trainalg == 2:
        algorithm = 'Gausskernel'    
    elif trainalg == 3:
        algorithm = 'NNet(Backprop)'
    elif trainalg == 4:
        algorithm =  'NNet(Congrad)'
    elif trainalg == 5:
        algorithm =  'NNet(Kalman)'    
    elif trainalg == 6:
        algorithm =  'Dnn(tensorflow)'    
    else:
        algorithm = 'SVM'    
    print 'Training with %s'%algorithm          
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
#  output files
    path = os.path.dirname(infile)
    basename = os.path.basename(infile)
    root, ext = os.path.splitext(basename)
    outfile = '%s/%s_class%s'%(path,root,ext)  
    tstfile = '%s/%s_%s.tst'%(path,root,algorithm)            
    if (trainalg in (2,3,4,5,6)) and probs:
#      class probabilities file
        probfile = '%s/%s_classprobs%s'%(path,root,ext) 
    else:
        probfile = None        
        
#  get the training data        
    Xs,Ls,K,classnames = rs.readshp(trnfile,inDataset,pos) 
    m = Ls.shape[0]  
#  stretch the pixel vectors to [-1,1] for ffn, dnn
    maxx = np.max(Xs,0)
    minx = np.min(Xs,0)
    for j in range(len(pos)):
        Xs[:,j] = 2*(Xs[:,j]-minx[j])/(maxx[j]-minx[j]) - 1.0 
#  random permutation of training data
    idx = np.random.permutation(m)
    Xs = Xs[idx,:] 
    Ls = Ls[idx,:]     
#  train on trainfrac of training examples, rest for testing          
    Xstrn = Xs[:int(trainfrac*m),:]
    Lstrn = Ls[:int(trainfrac*m),:] 
    Xstst = Xs[int(trainfrac*m):,:]  
    Lstst = Ls[int(trainfrac*m):,:]  
        
#  setup output datasets 
    driver = inDataset.GetDriver() 
    outDataset = driver.Create(outfile,cols,rows,1,GDT_Byte) 
    projection = inDataset.GetProjection()
    if geotransform is not None:
        outDataset.SetGeoTransform(geotransform)
    if projection is not None:
        outDataset.SetProjection(projection) 
    outBand = outDataset.GetRasterBand(1) 
    if probfile:   
        probDataset = driver.Create(probfile,cols,rows,K,GDT_Byte) 
        if geotransform is not None:
            probDataset.SetGeoTransform(geotransform)
        if projection is not None:
            probDataset.SetProjection(projection)  
        probBands = [] 
        for k in range(K):
            probBands.append(probDataset.GetRasterBand(k+1))         
#  initialize classifier  
    if   trainalg == 1:
        classifier = sc.Maxlike(Xstrn,Lstrn)
    elif trainalg == 2:
        classifier = sc.Gausskernel(Xstrn,Lstrn)
    elif trainalg == 3:
        classifier = sc.Ffnbp(Xstrn,Lstrn,L,epochs,validation)
    elif trainalg == 4:
        classifier = sc.Ffncg(Xstrn,Lstrn,L,epochs,validation)
    elif trainalg == 5:
        classifier = sc.Ffnekf(Xstrn,Lstrn,L,epochs,validation)    
    elif trainalg == 6:
#        classifier = sc.Dnn_learn(Xstrn,Lstrn,L,epochs) 
#        classifier = sc.Dnn_core(Xstrn,Lstrn,L,epochs)
        classifier = sc.Dnn_keras(Xstrn,Lstrn,L,epochs)
    elif trainalg == 7:
        classifier = sc.Svm(Xstrn,Lstrn)         
#  train it            
    print 'training on %i pixel vectors...' % np.max(classifier._Gs.shape)
    print 'classes: %s'%str(classnames)
    start = time.time()
    result = classifier.train()
    print 'elapsed time %s' %str(time.time()-start) 
    if result is not None:
        if (trainalg in [3,4,5]) and graphics:
#          the cost arrays are returned in result         
            cost = np.log(result[0]) 
            costv = np.log(result[1])
            ymax = np.max(cost)
            #ymin = np.min(cost)-1
            ymin = 5.0
            xmax = len(cost)      
            plt.plot(range(xmax),costv,'r',range(xmax),cost,'b')
            plt.axis([0,xmax,ymin,ymax])
            plt.title('Log(Cross entropy)')
            plt.xlabel('Epoch')              
#      classify the image           
        print 'classifying...'
        start = time.time()
        tile = np.zeros((outbuffer*cols,N),dtype=np.float32)    
        for row in range(rows/outbuffer):
            print 'row: %i'%(row*outbuffer)
            for j in range(N):
                tile[:,j] = rasterBands[j].ReadAsArray(0,row*outbuffer,cols,outbuffer).ravel()
                tile[:,j] = 2*(tile[:,j]-minx[j])/(maxx[j]-minx[j]) - 1.0               
            cls, Ms = classifier.classify(tile)  
            outBand.WriteArray(np.reshape(cls,(outbuffer,cols)),0,row*outbuffer)
            if probfile and Ms is not None:
                Ms = np.byte(Ms*255)
                for k in range(K):
                    probBands[k].WriteArray(np.reshape(Ms[:,k],(outbuffer,cols)),0,row*outbuffer)
        outBand.FlushCache()
        print 'elapsed time %s' %str(time.time()-start)
        outDataset = None
        inDataset = None      
        if probfile:
            for probBand in probBands:
                probBand.FlushCache() 
            probDataset = None
            print 'class probabilities written to: %s'%probfile                       
        print 'thematic map written to: %s'%outfile
        if (trainalg in [3,4,5]) and graphics:
            plt.show()
        if tstfile:
            with open(tstfile,'w') as f:               
                print >>f, algorithm +'test results for %s'%infile
                print >>f, time.asctime()
                print >>f, 'Classification image: %s'%outfile
                print >>f, 'Class probabilities image: %s'%probfile
                print >>f, Lstst.shape[0],Lstst.shape[1]
                classes, _ = classifier.classify(Xstst)
                labels = np.argmax(Lstst,axis=1)+1
                for i in range(len(classes)):
                    print >>f, classes[i], labels[i]              
                f.close()
                print 'test results written to: %s'%tstfile
        print 'done'
    else:
        print 'an error occured' 
        return 
   
if __name__ == '__main__':
    main()