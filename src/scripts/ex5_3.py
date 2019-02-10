#!/usr/bin/env python
#Name:  ex5_3.py

import  sys
import numpy as np

def parse_gcp(gcpfile):
    with open(gcpfile) as f:
        pts = []
        for i in range(6):
            line =f.readline()
        while line:
            pts.append(map(eval,line.split()))
            line = f.readline()
        f.close()    
    pts = np.array(pts)     
    return (pts[:,:2],pts[:,2:])  
    
def main(): 
    infile = sys.argv[1]  # gcps
    if infile:
        pts1,pts2 = parse_gcp(infile)
    else:
        return
    n = len(pts1)    
    y = pts1.ravel()
    A = np.zeros((2*n,4))
    for i in range(n):
        A[2*i,:] =   [pts2[i,0],-pts2[i,1],1,0]
        A[2*i+1,:] = [pts2[i,1], pts2[i,0],0,1]  
    result = np.linalg.lstsq(A,y,rcond=-1)    
    a,b,x0,y0 = result[0]
    RMS = np.sqrt(result[1]/n)
    print 'RST transformation:'
    print np.array([[a,-b,x0],[b,a,y0],[0,0,1]]) 
    print 'RMS: %f'%RMS
   
if __name__ == '__main__':
    main()    