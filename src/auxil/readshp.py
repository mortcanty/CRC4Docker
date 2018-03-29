#!/usr/bin/env python
#  Name: readshp.py
#  Purpose:
#     Read ENVI ROI shapefiles and return training/test data and class labels
#  Usage:    
#     import readshp
#     Xs, Ls, numclasses  = readshp.readshp(<train shapefile>, <inDataset for image>, <band positions)> )

import ogr, osr
from shapely.geometry import asPolygon, MultiPoint
import shapely.wkt  
import numpy as np

def readshp(trnfile,inDataset,pos):
#  projection info from input image  
    projection = inDataset.GetProjection()
    geotransform = inDataset.GetGeoTransform()
    gt = list(geotransform) 
    imsr = osr.SpatialReference()  
    imsr.ImportFromWkt(projection) 

    trnDriver = ogr.GetDriverByName('ESRI Shapefile')
    trnDatasource = trnDriver.Open(trnfile,0)
    trnLayer = trnDatasource.GetLayer() 
    trnsr = trnLayer.GetSpatialRef()             
#  coordinate transformation from training to image projection   
    ct = osr.CoordinateTransformation(trnsr,imsr) 
#  image bands     
    rasterBands = [] 
    for b in pos:
        rasterBands.append(inDataset.GetRasterBand(b)) 
#  number of classes    
    K = 1
    feature = trnLayer.GetNextFeature() 
    while feature:
        classid = feature.GetField('CLASS_ID')
        if int(classid)>K:
            K = int(classid)
        feature = trnLayer.GetNextFeature() 
    trnLayer.ResetReading()    
    K += 1          
#  loop through the polygons    
    Xs = []   # train observations (data matrix)
    Ls = []   # class labels (lists)
    classnames = []
    classids = set()
    print 'reading training data...'
    for i in range(trnLayer.GetFeatureCount()):
        feature = trnLayer.GetFeature(i)
        classid = str(feature.GetField('CLASS_ID'))
        classname  = feature.GetField('CLASS_NAME')
        if classid not in classids:
            classnames.append(classname)
        classids = classids | set(classid)     
#      label for this ROI           
        y = int(classid)
        l = [0 for i in range(K)]
        l[y] = 1.0
        polygon = feature.GetGeometryRef()
#      transform to same projection as image        
        polygon.Transform(ct)  
#      convert to a Shapely object            
        poly = shapely.wkt.loads(polygon.ExportToWkt())
#      transform the boundary to pixel coords in numpy        
        bdry = np.array(poly.boundary) 
        bdry[:,0] = bdry[:,0]-gt[0]
        bdry[:,1] = bdry[:,1]-gt[3]
        GT = np.mat([[gt[1],gt[2]],[gt[4],gt[5]]])
        bdry = bdry*np.linalg.inv(GT) 
#      polygon in pixel coords        
        polygon1 = asPolygon(bdry)
#      raster over the bounding rectangle        
        minx,miny,maxx,maxy = map(int,list(polygon1.bounds))  
        pts = [] 
        for i in range(minx,maxx+1):
            for j in range(miny,maxy+1): 
                pts.append((i,j))             
        multipt =  MultiPoint(pts)   
#      intersection as list              
        intersection = np.array(multipt.intersection(polygon1),dtype=np.int).tolist()
#      cut out the bounded image cube               
        cube = np.zeros((maxy-miny+1,maxx-minx+1,len(rasterBands)))
        k=0
        for band in rasterBands:
            cube[:,:,k] = band.ReadAsArray(minx,miny,maxx-minx+1,maxy-miny+1)
            k += 1
#      get the training vectors
        for (x,y) in intersection:         
            Xs.append(cube[y-miny,x-minx,:])
            Ls.append(l)   
        polygon = None
        polygon1 = None            
        feature.Destroy()  
    trnDatasource.Destroy()       
    Xs = np.array(Xs) 
    Ls = np.array(Ls)   
    return (Xs,Ls,K,classnames)         
    
if __name__ == '__main__':
    pass
    