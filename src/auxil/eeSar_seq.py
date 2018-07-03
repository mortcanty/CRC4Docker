'''
Created on 21.06.2018

@author: mort
'''
import ee, math, time
import ipywidgets as widgets
from IPython.display import display
from ipyleaflet import (Map,DrawControl,TileLayer,basemaps,basemap_to_tiles,SplitMapControl)
from auxil.eeWishart import omnibus

ee.Initialize()

poly = ee.Geometry.Polygon([[6.30154, 50.948329], [6.293307, 50.877329], 
                            [6.427091, 50.875595], [6.417486, 50.947464], 
                            [6.30154, 50.948329]])
center = list(reversed(poly.centroid().coordinates().getInfo()))

def get_vvvh(image):
    ''' get 'VV' and 'VH' bands from sentinel-1 imageCollection and restore linear signal from db-values '''
    return image.select('VV','VH').multiply(ee.Image.constant(math.log(10.0)/10.0)).exp()

def get_image(current,image):
    ''' accumulate a single image from a collection of images '''
    return ee.Image.cat(ee.Image(image),current)    
    
def clipList(current,prev):
    ''' clip a list of images '''
    imlist = ee.List(ee.Dictionary(prev).get('imlist'))
    poly = ee.Dictionary(prev).get('poly')    
    imlist = imlist.add(ee.Image(current).clip(poly))
    return ee.Dictionary({'imlist':imlist,'poly':poly})

def handle_draw(self, action, geo_json):
    global poly
    if action == 'created':
        coords =  geo_json['geometry']['coordinates']
        poly = ee.Geometry.Polygon(coords)
        
dc = DrawControl()
dc.on_draw(handle_draw)

def GetTileLayerUrl(ee_image_object):
    map_id = ee.Image(ee_image_object).getMapId()
    tile_url_template = "https://earthengine.googleapis.com/map/{mapid}/{{z}}/{{x}}/{{y}}?token={token}"
    return tile_url_template.format(**map_id)

w_orbitpass = widgets.RadioButtons(
    options=['ASCENDING','DESCENDING'],
     value='ASCENDING',
    description='Orbit pass:',
    disabled=False
)
w_changemap = widgets.RadioButtons(
    options=['First','Last','Frequency'],
     value='First',
    description='Change map:',
    disabled=False
)
w_relativeorbitnumber = widgets.IntText(
    value=0,
    description='Rel orbit:',
    disabled=False
)
w_exportname = widgets.Text(
    value='users/mortcanty/omnibus/trail',
    placeholder=' ',
    description='Asset:',
    disabled=False
)
w_startdate = widgets.Text(
    value='2017-04-01',
    placeholder=' ',
    description='Start date:',
    disabled=False
)
w_enddate = widgets.Text(
    value='2017-10-01',
    placeholder=' ',
    description='End date:',
    disabled=False
)
w_median = widgets.Checkbox(
    value=True,
    description='Median filter',
    disabled=False
)
w_significance = widgets.BoundedFloatText(
    value=0.0001,
    min=0,
    max=0.05,
    step=0.0001,
    description='Significance:',
    disabled=False
)
w_opacity = widgets.BoundedFloatText(
    value=0.5,
    min=0.0,
    max=1.0,
    step=0.1,
    description='Opacity:',
    disabled=False
)



w_run = widgets.Button(description="Run")
w_preview = widgets.Button(description="Preview")
w_export = widgets.Button(description='Export')
w_dates = widgets.HBox([w_startdate,w_enddate])
w_orbit = widgets.HBox([w_relativeorbitnumber,w_orbitpass,w_changemap,w_opacity])
w_exp = widgets.HBox([w_export,w_exportname])
w_signif = widgets.HBox([w_significance,w_median])
w_rse = widgets.HBox([w_run,w_exp,w_preview])

box = widgets.VBox([w_dates,w_orbit,w_signif,w_rse])

def on_run_button_clicked(b):
    global result,collection,count,timestamplist1, \
           w_startdate,w_enddate,w_orbitpass,w_changemap, \
           w_relativeorbitnumber,w_significance,w_median
    try:
        collection = ee.ImageCollection('COPERNICUS/S1_GRD') \
                  .filterBounds(poly) \
                  .filterDate(ee.Date(w_startdate.value), ee.Date(w_enddate.value)) \
                  .filter(ee.Filter.eq('transmitterReceiverPolarisation', ['VV','VH'])) \
                  .filter(ee.Filter.eq('resolution_meters', 10)) \
                  .filter(ee.Filter.eq('instrumentMode', 'IW')) \
                  .filter(ee.Filter.eq('orbitProperties_pass', w_orbitpass.value))   
        if w_relativeorbitnumber.value > 0:
            collection = collection.filter(ee.Filter.eq('relativeOrbitNumber_start', int(w_relativeorbitnumber.value)))            
        collection = collection.sort('system:time_start') 

        acquisition_times = ee.List(collection.aggregate_array('system:time_start')).getInfo()
        count = len(acquisition_times) 
        if count<2:
            raise ValueError('Less than 2 images found')
        timestamplist = []
        for timestamp in acquisition_times:
            tmp = time.gmtime(int(timestamp)/1000)
            timestamplist.append(time.strftime('%x', tmp))  
    #  make timestamps in YYYYMMDD format            
        timestamplist = [x.replace('/','') for x in timestamplist]  
        timestamplist = ['T20'+x[4:]+x[0:4] for x in timestamplist]
    #  in case of duplicates add running integer
        timestamplist1 = [timestamplist[i] + '_' + str(i+1) for i in range(len(timestamplist))]    
        relativeorbitnumbers = str(map(int,ee.List(collection.aggregate_array('relativeOrbitNumber_start')).getInfo()))
        print 'Images found: %i'%count
        print 'Acquisition dates: '+timestamplist[0]+'...'+timestamplist[-1]
        print 'Relative orbit numbers: '+relativeorbitnumbers[0:40]+'...'
        pcollection = collection.map(get_vvvh)
        pList = pcollection.toList(100)   
        first = ee.Dictionary({'imlist':ee.List([]),'poly':poly}) 
        imList = ee.Dictionary(pList.iterate(clipList,first)).get('imlist')
        result = ee.Dictionary(omnibus(imList,w_significance.value,w_median.value))
    except Exception as e:
        print 'An error occurred in Omnibus: %s'%e

w_run.on_click(on_run_button_clicked)

def on_preview_button_clicked(b):
    global result,m,cmap,smap,fmap,bmap,w_changemap
    jet = 'black,blue,cyan,yellow,red'
    smap = ee.Image(result.get('smap')).byte()
    cmap = ee.Image(result.get('cmap')).byte()
    fmap = ee.Image(result.get('fmap')).byte() 
    bmap = ee.Image(result.get('bmap')).byte() 
    opacity = w_opacity.value
    if w_changemap.value=='First':
        mp = smap 
        mx = count
    elif w_changemap.value=='Last':
        mp=cmap
        mx = count
    else:
        mp = fmap
        mx = count/2
    if len(m.layers)>1:
        m.remove_layer(m.layers[1])
    m.add_layer(TileLayer(url=GetTileLayerUrl( mp.visualize(min=0, max=mx, palette=jet,opacity = opacity))))
    
w_preview.on_click(on_preview_button_clicked)   

def on_export_button_clicked(b):
    global w_exportname
    collection1 = ee.ImageCollection('COPERNICUS/S2') \
                    .filterBounds(poly) \
                    .filterDate(ee.Date(w_startdate.value),ee.Date(w_enddate.value)) \
                    .sort('CLOUDY_PIXEL_PERCENTAGE',True) \
                    .filterMetadata('CLOUDY_PIXEL_PERCENTAGE','less_than',1.0) 
    count1 = collection1.size().getInfo()
    if count1>0:
#      use sentinel-2 as background                        
        background = ee.Image(collection1.first()) \
                               .clip(poly) \
                               .select('B8') \
                               .divide(5000)    
    else:
#      use temporal averaged sentinel-1        
        background = collection.mean() \
                               .select(0) \
                               .clip(poly) \
                               .divide(20) \
                               .add(1)                                                
    background = background.where(background.gte(1),1) \
                           .where(background.lte(0),0)   
    cmaps = ee.Image.cat(cmap,smap,fmap,bmap,background).rename(['cmap','smap','fmap']+timestamplist1[1:]+['background'])                
    assexport = ee.batch.Export.image.toAsset(cmaps,
                                description='assetExportTask', 
                                assetId=w_exportname.value,scale=10,maxPixels=1e9)
    assexportid = str(assexport.id)
    print '****Exporting to %s in Assets, task id: %s'%(w_exportname.value,assexportid)
    assexport.start()  
    
w_export.on_click(on_export_button_clicked)  

def run():
    global m,dc,center
    center = list(reversed(poly.centroid().coordinates().getInfo()))
    m = Map(center=center, zoom=11, layout={'height':'400px'})
    osm = basemap_to_tiles(basemaps.OpenStreetMap.HOT)
    mb = TileLayer(url="https://api.mapbox.com/styles/v1/mapbox/satellite-streets-v9/tiles/256/{z}/{x}/{y}?access_token=pk.eyJ1IjoibWNhbnR5IiwiYSI6ImNpcjRsMmJxazAwM3hoeW05aDA1cmNkNzMifQ.d2UbIugbQFk2lnU8uHwCsQ")
    sm_control = SplitMapControl(left_layer=osm,right_layer=mb)
    m.add_control(dc)
    m.add_control(sm_control)
    display(m)
    display(box)