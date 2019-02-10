#!/usr/bin/env python
#******************************************************************************
#  Name:     mcnemar.py
#  Purpose:  compare two classifiers with McNemar statistic
#  Usage:             
#    python mcnenmar.py
#
#  Copyright (c) 2018, Mort Canty


import auxil.auxil1 as auxil
import numpy as np
from scipy import stats 
import sys, getopt

def main():
    usage = '''
    Usage: 
    python %s testfile1 testfile2''' %sys.argv[0]
    options, args = getopt.getopt(sys.argv[1:],'h')
    for option, _ in options:
        if option == '-h':
            print usage
            return                     
    if len(args) != 2: 
        print 'Incorrect number of arguments'
        print usage
        sys.exit(1)          
    tstfile1 = args[0]
    tstfile2 = args[1]
    print '========================='
    print '     McNemar test'
    print '========================='
    with open(tstfile1,'r') as f1:
        with open(tstfile2,'r') as f2:
            line = ''
            for i in range(4):
                line += f1.readline()
            print 'first classifier:\n'+line    
            line = f1.readline().split()
            n1 = int(line[0]) 
            K1 = int(line[1]) 
            line = ''               
            for i in range(4):
                line += f2.readline()
            print 'second classifier:\n'+line   
            line = f2.readline().split()    
            n2 = int(line[0]) 
            K2 = int(line[1])
            if (n1 != n2) or (K1 != K2):
                print 'test files are incompatible'
                return
            print 'test observations: %i'%n1
            print 'classes: %i'%K1
#          calculate McNemar
            y10 = 0.0
            y01 = 0.0
            for i in range(n1):
                line = f1.readline()
                k = map(int,line.split())
                k1A = k[0]
                k2A = k[1]
                line = f2.readline()
                k = map(int,line.split())
                k1B = k[0]
                k2B = k[1]
                if (k1A != k2A) and (k1B == k2B):
                    y10 += 1
                if (k1A == k2A) and (k1B != k2B):
                    y01 += 1        
    f1.close()
    f2.close()
    McN = (np.abs(y01-y10))**2/(y10+y01)
    print 'first classifier: %i'%int(y10)      
    print 'second classifier: %i'%int(y01)
    print 'McNemar statistic: %f'%McN
    print 'P-value: %f'%(1-stats.chi2.cdf(McN,1))
          
if __name__ == '__main__':
    main()     