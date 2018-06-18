#!/usr/bin/env python
#******************************************************************************
#  Name:     ct.py
#  Purpose:  determine classification accuracy and contingency table
#            from test data
#  Usage:             
#    python ct.py 
#
#  Copyright (c) 2018, Mort Canty

import numpy as np
import contextlib 
import sys, getopt

@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    yield 
    np.set_printoptions(**original)

def main():
    usage = '''
    Usage: 
    python %s testfile ''' %sys.argv[0]
    options, args = getopt.getopt(sys.argv[1:],'h')
    for option, _ in options:
        if option == '-h':
            print usage
            return                     
    if len(args) != 1: 
        print 'Incorrect number of arguments'
        print usage
        sys.exit(1)          
    tstfile = args[0]
    if not tstfile:
        return
    print '========================='
    print 'classification statistics'
    print '========================='
    with open(tstfile,'r') as f:
        line = ''
        for i in range(4):
            line += f.readline()
        print line    
        line = f.readline().split()
        n = int(line[0]) 
        K = int(line[1])
        CT = np.zeros((K+2,K+2))
#      fill the contingency table
        y = 0.0
        line = f.readline()
        while line:
            k = map(int,line.split())
            k1 = k[0]-1
            k2 = k[1]-1
            CT[k1,k2] += 1
            if k1 != k2:
                y += 1
            line = f.readline()
        f.close()
        CT[K,:] = np.sum(CT, axis=0)
        CT[:,K] = np.sum(CT, axis=1)
        for i in range(K):
            CT[K+1,i] = CT[i,i]/CT[K,i]
            CT[i,K+1] = CT[i,i]/CT[i,K]      
#      overall misclassification rate
        sigma = np.sqrt(y*(n-y)/n**3)
        low = (y+1.921-1.96*np.sqrt(0.96+y*(n-y)/n))/(3.842+n)
        high= (y+1.921+1.96*np.sqrt(0.96+y*(n-y)/n))/(3.842+n)
        print 'Misclassification rate: %f'%(y/n)
        print 'Standard deviation: %f'%sigma
        print 'Conf. interval (95 percent): [%f , %f]'%(low, high)
#      Kappa coefficient
        t1 = float(n-y)/n
        t2 = np.sum(CT[K,0:K]*np.transpose(CT[0:K,K]))/n**2
        Kappa = (t1 - t2)/(1 - t2)
        t3 = 0.0
        for i in range(K):
            t3 = t3 + CT[i,i]*(CT[K,i]+CT[i,K])
        t3 = t3/n**2
        t4 = 0.0
        for i in range(K):
            for j in range(K):
                t4 += CT[j,i]*(CT[K,j]+CT[i,K])**2
        t4 = t4/n**3
        sigma2 = t1*(1-t1)/(1-t2)**2
        sigma2 = sigma2 + 2*(1-t1)*(2*t1*t2-t3)/(1-t2)**3
        sigma2 = sigma2 + ((1-t1)**2)*(t4-4*t2**2)/(1-t2)**4
        sigma = np.sqrt(sigma2/n)
        print 'Kappa coefficient: %f'%Kappa
        print 'Standard deviation: %f'%sigma
        print 'Contingency Table'
        with printoptions(precision=3, linewidth = 200, suppress=True):
            print CT
         
if __name__ == '__main__':
    main()     