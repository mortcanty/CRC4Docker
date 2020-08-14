#!/usr/bin/env python
#******************************************************************************
#  Name:     supervisedclass.py
#  Purpose:  object classes for supervised image classification, maximum likelihood, 
#            Gaussian kernel, feed forward nn (back-propagation,scaled conjugate gradient, 
#            extended Kalman filter), deep learning nn(tensorflow), support vector machine
#  Usage:    
#     import supervisedclass
#
# (c) Mort Canty 2018

import auxil1
import numpy as np  
import tensorflow as tf 
from tensorflow.keras import layers
from scipy.optimize import minimize_scalar
from mlpy import MaximumLikelihoodC, LibSvm   

tf.logging.set_verbosity('ERROR')   

class Maxlike(MaximumLikelihoodC):
       
    def __init__(self,Gs,ls): 
        MaximumLikelihoodC.__init__(self)
        self._K = ls.shape[1] 
        self._Gs = Gs  
        self._N = Gs.shape[1]
        self._ls = ls
          
    def train(self):
        try: 
            labels = np.argmax(self._ls,axis=1)
            idx = np.where(labels == 0)[0]
            ls = np.ones(len(idx),dtype=np.int)
            Gs = self._Gs[idx,:]
            for k in range(1,self._K):
                idx = np.where(labels == k)[0]
                ls = np.concatenate((ls, \
                (k+1)*np.ones(len(idx),dtype=np.int)))
                Gs = np.concatenate((Gs,\
                               self._Gs[idx,:]),axis=0)         
            self.learn(Gs,ls)    
            return True 
        except Exception as e:
            print 'Error: %s'%e 
            return None    
              
    def classify(self,Gs):
        classes = self.pred(Gs)
        return (classes, None)  
    
    def test(self,Gs,ls):
        m = np.shape(Gs)[0]
        classes, _ = self.classify(Gs)
        classes = np.asarray(classes,np.int16)
        labels = np.argmax(np.transpose(ls),axis=0)+1
        misscls = np.where(classes-labels)[0]
        return len(misscls)/float(m)
    
class Gausskernel(object):
    
    def __init__(self,Gs,ls): 
        self._K = ls.shape[1] 
        self._Gs = Gs 
        self._N = Gs.shape[1]
        self._ls = np.argmax(ls,1)
        self._m = Gs.shape[0]
        
    def output(self,sigma,Hs,symm=True):
        pvs = np.zeros((Hs.shape[0],self._K))
        kappa = auxil1.kernelMatrix(
            Hs,self._Gs,gma=0.5/(sigma**2),
                                nscale=1,kernel=1)[0]
        if symm:
            kappa[range(self._m),range(self._m)] = 0
        for j in range(self._K):
            kpa = np.copy(kappa)            
            idx = np.where(self._ls!=j)[0]
            nj = self._m - idx.size
            kpa[:,idx] = 0
            pvs[:,j] = np.sum(kpa,1).ravel()/nj
        s = np.transpose(np.tile(np.sum(pvs,1),
                                   (self._K,1)))                        
        return pvs/s
    
    def theta(self,sigma):  
        pvs = self.output(sigma,self._Gs,True)   
        labels = np.argmax(pvs,1)
        idx = np.where(labels != self._ls)[0]
        n = idx.size
        error = float(n)/(self._m)
        print 'sigma: %f  error: %f'%(sigma,error)
        return error
    
    def train(self):        
        result = minimize_scalar(
         self.theta,bracket=(0.001,0.1,1.0),tol=0.001)
        if result.success:
            self._sigma_min = result.x
            return True 
        else:
            print result.message
            return None
        
    def classify(self,Gs): 
        pvs = self.output(self._sigma_min,Gs,False)   
        classes = np.argmax(pvs,1)+1
        return (classes,pvs)    
    
    def test(self,Gs,ls):
        m = np.shape(Gs)[0]
        classes, _ = self.classify(Gs)
        classes = np.asarray(classes,np.int16)
        labels = np.argmax(np.transpose(ls),axis=0)+1
        misscls = np.where(classes-labels)[0]
        return len(misscls)/float(m)    
        

class Ffn(object):
 
    def __init__(self,Gs,ls,Ls,epochs,validate): 
#      setup the network architecture        
        self._L = Ls[0] 
        self._m,self._N = Gs.shape 
        self._K = ls.shape[1]
        self._epochs = epochs
#      biased input as column vectors         
        Gs = np.mat(Gs).T 
        self._Gs = np.vstack((np.ones(self._m),Gs))      
#      biased output vector from hidden layer        
        self._n = np.mat(np.zeros(self._L+1))         
#      labels as column vectors
        self._ls = np.mat(ls).T
        if validate:
#          split into train and validate sets 
            self._m = self._m/2          
            self._Gsv = self._Gs[:,self._m:]
            self._Gs = self._Gs[:,:self._m]
            self._lsv = self._ls[:,self._m:]
            self._ls = self._ls[:,:self._m]
        else:
            self._Gsv = self._Gs
            self._lsv = self._ls        
#      weight matrices
        self._Wh=np.mat(np.random. \
                      random((self._N+1,self._L)))-0.5
        self._Wo=np.mat(np.random. \
                      random((self._L+1,self._K)))-0.5      
                               
    def forwardpass(self,G):
#      forward pass through the network
        expnt = self._Wh.T*G
        self._n = np.vstack((np.ones(1),1.0/ \
                                  (1+np.exp(-expnt))))
#      softmax activation
        I = self._Wo.T*self._n
        A = np.exp(I-max(I))
        return A/np.sum(A)
    
    def classify(self,Gs):  
#      vectorized classes and membership probabilities
        Gs = np.mat(Gs).T
        m = Gs.shape[1]
        Gs = np.vstack((np.ones(m),Gs))
        expnt = self._Wh.T*Gs
        expnt[np.where(expnt<-100.0)] = -100.0
        expnt[np.where(expnt>100.0)] = 100.0        
        n=np.vstack((np.ones(m),1/(1+np.exp(-expnt))))
        Io = self._Wo.T*n
        maxIo = np.max(Io,axis=0)
        for k in range(self._K):
            Io[k,:] -= maxIo
        A = np.exp(Io)
        sm = np.sum(A,axis=0)
        Ms = np.zeros((self._K,m))
        for k in range(self._K):
            Ms[k,:] = A[k,:]/sm
        classes = np.argmax(Ms,axis=0)+1 
        return (classes, np.transpose(Ms))   
    
    def vforwardpass(self,Gs):
#      vectorized forward pass, Gs are biased column vectors
        m = Gs.shape[1]
        expnt = self._Wh.T*Gs
        n = np.vstack((np.ones(m),1.0/(1+np.exp(-expnt))))
        Io = self._Wo.T*n
        maxIo = np.max(Io,axis=0)
        for k in range(self._K):
            Io[k,:] -= maxIo
        A = np.exp(Io)
        sm = np.sum(A,axis=0) 
        Ms = np.zeros((self._K,m)) 
        for k in range(self._K):
            Ms[k,:] = A[k,:]/sm
        return (Ms, n)        
    
    def cost(self):
        Ms, _ = self.vforwardpass(self._Gs)
        return -np.sum(np.multiply(self._ls,np.log(Ms+1e-20)))
    
    def costv(self):
        Ms, _ = self.vforwardpass(self._Gsv)
        return -np.sum(np.multiply(self._lsv,np.log(Ms+1e-20)))

    def test(self,Gs,ls):
        m = np.shape(Gs)[0]
        classes, _ = self.classify(Gs)
        classes = np.asarray(classes,np.int16)
        labels = np.argmax(np.transpose(ls),axis=0)+1
        misscls = np.where(classes-labels)[0]
        return len(misscls)/float(m)    
    
class Ffnbp(Ffn):
    
    def __init__(self,Gs,ls,Ls,epochs=100,valid=False):
        Ffn.__init__(self,Gs,ls,Ls,epochs,valid)
           
    def train(self):
        eta = 0.01
        alpha = 0.5
        maxitr = self._epochs*self._m 
        inc_o1 = 0.0
        inc_h1 = 0.0
        epoch = 0
        cost = []
        costv = []
        itr = 0        
        try:
            while itr<maxitr:
#              select train example pair at random
                nu = np.random.randint(0,self._m)
                x = self._Gs[:,nu]
                ell = self._ls[:,nu]
#              send it through the network
                m = self.forwardpass(x)
#              determine the deltas
                d_o = ell - m
                d_h = np.multiply(np.multiply(self._n,\
                     (1-self._n)),(self._Wo*d_o))[1::]
#              update synaptic weights
                inc_o = eta*(self._n*d_o.T)
                inc_h = eta*(x*d_h.T)
                self._Wo += inc_o + alpha*inc_o1
                self._Wh += inc_h + alpha*inc_h1
                inc_o1 = inc_o
                inc_h1 = inc_h
#              record cost function
                if itr % self._m == 0:
                    cost.append(self.cost())
                    costv.append(self.costv())
                    epoch += 1
                itr += 1
        except Exception as e:
            print 'Error: %s'%e
            return None
        return (np.array(cost),np.array(costv))
    
class Ffncg(Ffn):
    
    def __init__(self,Gs,ls,Ls,epochs=100,validate=False):
        Ffn.__init__(self,Gs,ls,Ls,epochs,validate)
    
    def gradient(self):
#      gradient of cross entropy wrt synaptic weights          
        M,n = self.vforwardpass(self._Gs)
        D_o = self._ls - M
        D_h = np.mat(n.A*(1-n.A)*(self._Wo*D_o).A)[1::,:]
        dEh = -(self._Gs*D_h.T).ravel()
        dEo = -(n*D_o.T).ravel()
        return np.append(dEh.A,dEo.A)  
    
    def hessian(self):    
#      Hessian of cross entropy wrt synaptic weights        
        nw = self._L*(self._N+1)+self._K*(self._L+1)  
        v = np.eye(nw,dtype=np.float)  
        H = np.zeros((nw,nw))
        for i in range(nw):
            H[i,:] = self.rop(v[i,:])
        return H    
            
    def rop(self,V):     
#      reshape V to dimensions of Wh and Wo, transpose
        VhT = np.reshape(V[:(self._N+1)*self._L],
                         (self._N+1,self._L)).T
        Vo = np.mat(np.reshape(V[self._L*(self._N+1)::],
                         (self._L+1,self._K)))
        VoT = Vo.T
#      transpose the output weights
        Wo = self._Wo
        WoT = Wo.T 
#      forward pass
        M,n = self.vforwardpass(self._Gs) 
#      evaluation of v^T.H
        Z = np.zeros(self._m)  
        D_o = self._ls - M                 #d^o
        RIh = VhT*self._Gs                 #Rv{I^h}
        tmp = np.vstack((Z,RIh))         
        RN = n.A*(1-n.A)*tmp.A             #Rv{n}   
        RIo = WoT*RN + VoT*n               #Rv{I^o}
        Rd_o = -np.mat(M*(1-M)*RIo.A)      #Rv{d^o}
        Rd_h = n.A*(1-n.A)*( (1-2*n.A)*tmp.A
               *(Wo*D_o).A + (Vo*D_o).A + (Wo*Rd_o).A)
        Rd_h = np.mat(Rd_h[1::,:])         #Rv{d^h}
        REo = -(n*Rd_o.T-RN*D_o.T).ravel() #Rv{dE/dWo} 
        REh = -(self._Gs*Rd_h.T).ravel()   #Rv{dE/dWh}
        return np.hstack((REo,REh))        #v^T.H
                                         
    def train(self):
        try: 
            cost = []   
            costv = []          
            w = np.concatenate((self._Wh.A.ravel(),
                                self._Wo.A.ravel()))
            nw = len(w)
            g = self.gradient()
            d = -g
            k = 0
            lam = 0.001
            while k < self._epochs:
                d2=np.sum(d*d)              # d^2
                dTHd=np.sum(self.rop(d).A*d)# d^T.H.d
                delta = dTHd + lam*d2
                if delta < 0:
                    lam = 2*(lam-delta/d2)
                    delta = -dTHd
                E1 = self.cost()            # E(w)
                dTg = np.sum(d*g)           # d^T.g
                alpha = -dTg/delta
                dw = alpha*d
                w += dw
                self._Wh = np.mat(np.reshape(
                       w[0:self._L*(self._N+1)], 
                       (self._N+1,self._L)))            
                self._Wo = np.mat(np.reshape(
                       w[self._L*(self._N+1)::],
                       (self._L+1,self._K)))
                E2 = self.cost()           # E(w+dw)
                Ddelta = -2*(E1-E2)/(alpha*dTg)
                if Ddelta < 0.25:
                    w -= dw                # undo 
                    self._Wh = np.mat(np.reshape(
                        w[0:self._L*(self._N+1)],
                        (self._N+1,self._L)))
                    self._Wo = np.mat(np.reshape(
                        w[self._L*(self._N+1)::],
                        (self._L+1,self._K)))     
                    lam *= 4.0      # decrease step
                    if lam > 1e20:  # step too small
                        k = self._epochs  # give up
                    else:          # else
                        d = -g     # restart             
                else:
                    k += 1
                    cost.append(E1)   
                    costv.append(self.costv())          
                    if Ddelta > 0.75:
                        lam /= 2.0
                    g = self.gradient()
                    if k % nw == 0:
                        beta = 0.0
                    else:
                        beta = np.sum(
                            self.rop(g).A*d)/dTHd
                    d = beta*d - g
            return (cost,costv) 
        except Exception as e:
            print 'Error: %s'%e
            return None     
        
class Ffnekf(Ffn):
    
    def __init__(self,Gs,ls,Ls,epochs=10,validate=False):
        Ffn.__init__(self,Gs,ls,Ls,epochs,validate)
#      weight covariance matrices
        self._Sh = np.zeros((self._N+1,self._N+1,self._L)) 
        for i in range(self._L):
            self._Sh[:,:,i] = np.identity(self._N+1)*100        
        self._So = np.zeros((self._L+1,self._L+1,self._K)) 
        for i in range(self._K):
            self._So[:,:,i] = np.identity(self._L+1)*100       
            
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
    #          select random training pair
                nu = np.random.randint(0,self._m)
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
    
class Dnn_learn(object):    
    '''High-level Dnn classifier, now deprecated '''
    def __init__(self,Gs,ls,Ls,epochs=1000):
#      setup the network architecture, Geron, p.164     
        self._Gs = Gs
        self._y = np.argmax(ls,1)
        self._epochs = epochs
        n_classes = ls.shape[1]
        feature_cols = tf.contrib.learn. \
        infer_real_valued_columns_from_input(self._Gs)
        dnn = tf.contrib.learn.DNNClassifier(   
                hidden_units=Ls, 
                n_classes=n_classes, 
                feature_columns=feature_cols)
        self._dnn=tf.contrib.learn.SKCompat(dnn)
        
    def train(self):
        try:
            self._dnn.fit(self._Gs,self._y,
                             steps=self._epochs)
            return True 
        except Exception as e:
            print 'Error: %s'%e 
            return None    
        
    def classify(self,Gs):
        result = self._dnn.predict(Gs)
        return (result['classes']+1, 
                result['probabilities'])    
        
    def test(self,Gs,ls):
        m = np.shape(Gs)[0]
        classes, _ = self.classify(Gs)
        classes = np.asarray(classes,np.int16)
        labels = np.argmax(np.transpose(ls),axis=0)+1
        misscls = np.where(classes-labels)[0]
        return len(misscls)/float(m)       
    
class Dnn_core(object):    
    '''High-level core TensorFlow Dnn classifier'''
    def __init__(self,Gs,ls,Ls,epochs=100):
#      setup the network architecture  
        self._Gs = Gs   
        n_classes = ls.shape[1]
        labels = np.argmax(ls,1)
        N = Gs.shape[1]
        self._epochs = epochs
#      feature column structure        
        feature_cols = [
         tf.feature_column.numeric_column('x',(1,N))
                       ]
#      input function for training                
        self._train_input_fn = tf.estimator.inputs\
           .numpy_input_fn(x={'x': Gs},
                           y=labels,
                           shuffle=True,
                           num_epochs = None,
                           batch_size = Gs.shape[0])
#      instantiate the classifier                    
        self._dnn = tf.estimator.DNNClassifier(   
                hidden_units=Ls, 
                n_classes=n_classes,
                feature_columns=feature_cols)
                    
    def train(self):
        try:
#          train the classifier            
            self._dnn.train(
                input_fn=self._train_input_fn,
                          steps=self._epochs)
            return True 
        except Exception as e:
            print 'Error: %s'%e 
            return None    
        
    def classify(self,Gs):
#      input function for prediction    
        predict_input_fn = tf.estimator.inputs\
                   .numpy_input_fn(x={'x': Gs},
                                     y=None,
                                     shuffle=False,
                                     num_epochs=1)
#      classify new data                   
        result = self._dnn.predict(predict_input_fn)
        classes = []; probabilities = []
        for r in result:
            classes.append(r['class_ids'][0])
            probabilities.append(r['probabilities'])
        return (np.array(classes)+1, 
                np.array(probabilities))

    def test(self,Gs,ls):
        m = np.shape(Gs)[0]
        classes, _ = self.classify(Gs)
        classes = np.asarray(classes,np.int16)
        labels = np.argmax(np.transpose(ls),axis=0)+1
        misscls = np.where(classes-labels)[0]
        return len(misscls)/float(m)               
    
class Dnn_keras(object):    
    '''High-level TensorFlow (keras) Dnn classifier'''
    def __init__(self,Gs,ls,Ls,epochs=100):
#      setup the network architecture  
        self._Gs = Gs   
        n_classes = ls.shape[1]
        self._labels = ls
        self._epochs = epochs
        self._dnn = tf.keras.Sequential()
#      hidden layers        
        for L in Ls:
            self._dnn \
             .add(layers.Dense(L,activation='relu'))
#      output layer
        self._dnn \
          .add(layers.Dense(n_classes,
                            activation='softmax'))       
#      initialize                             
        self._dnn.compile(
     optimizer=tf.train.GradientDescentOptimizer(0.01),
     loss='categorical_crossentropy')
        
    def train(self):
        try:           
            self._dnn.fit(self._Gs,self._labels,
                        epochs=self._epochs,verbose=0)
            return True 
        except Exception as e:
            print 'Error: %s'%e 
            return None             
        
    def classify(self,Gs):     
#      predict new data                       
        Ms = self._dnn.predict(Gs)
        cls = np.argmax(Ms,1)+1
        return (cls,Ms)

    def test(self,Gs,ls):
        m = np.shape(Gs)[0]
        classes, _ = self.classify(Gs)
        classes = np.asarray(classes,np.int16)
        labels = np.argmax(np.transpose(ls),axis=0)+1
        misscls = np.where(classes-labels)[0]
        return len(misscls)/float(m)  


class Svm(object):   
      
    def __init__(self,Gs,ls):
        self._K = ls.shape[1]
        self._Gs = Gs
        self._N = Gs.shape[1]
        self._ls = ls
        self._svm = LibSvm('c_svc','rbf',\
            gamma=1.0/self._N,C=100,probability=True)                
      
    def train(self):
        try:
            labels = np.argmax(self._ls,axis=1)
            idx = np.where(labels == 0)[0]
            ls = np.ones(len(idx),dtype=np.int)
            Gs = self._Gs[idx,:]
            for k in range(1,self._K):
                idx = np.where(labels == k)[0]
                ls = np.concatenate((ls, \
                (k+1)*np.ones(len(idx),dtype=np.int)))
                Gs = np.concatenate((Gs,\
                               self._Gs[idx,:]),axis=0)         
            self._svm.learn(Gs,ls)  
            return True 
        except Exception as e:
            print 'Error: %s'%e 
            return None   
      
    def classify(self,Gs):
        probs = np.transpose(self._svm. \
                             pred_probability(Gs))       
        classes = np.argmax(probs,axis=0)+1
        return (classes, np.transpose(probs)) 
              
    def test(self,Gs,ls):
        m = np.shape(Gs)[0]
        classes, _ = self.classify(Gs)
        classes = np.asarray(classes,np.int16)
        labels = np.argmax(np.transpose(ls),axis=0)+1
        misscls = np.where(classes-labels)[0]
        return len(misscls)/float(m)              
    
if __name__ == '__main__':
#  test on random data    
    Gs = 2*np.random.random((1000,3)) -1.0
    ls = np.zeros((1000,4))
    for l in ls:
        l[np.random.randint(0,4)]=1.0 
    cl = Gausskernel(Gs,ls)  
    if cl.train():
        classes, probabilities = cl.classify(Gs)
        print classes
    
    