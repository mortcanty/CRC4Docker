'''
Created on 08.04.2019

@author: mort

ipywidget interface to the GEE for IR-MAD

'''
import ee, time, warnings, math
import ipywidgets as widgets
from IPython.display import display
from ipyleaflet import (Map,DrawControl,TileLayer,basemaps,basemap_to_tiles,SplitMapControl)
from auxil.eeMad import imad,radcal

ee.Initialize()

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

poly = ee.Geometry.Polygon([[6.30154, 50.948329], [6.293307, 50.877329], 
                            [6.427091, 50.875595], [6.417486, 50.947464], 
                            [6.30154, 50.948329]])

center = list(reversed(poly.centroid().coordinates().getInfo()))

def chi2cdf(chi2,df):
    ''' Chi square cumulative distribution function '''
    return ee.Image(chi2.divide(2)).gammainc(ee.Number(df).divide(2))


def makefeature(data):
    ''' for exporting as CSV to Drive '''
    return ee.Feature(None, {'data': data})

def handle_draw(self, action, geo_json):
    global poly
    if action == 'created':
        coords =  geo_json['geometry']['coordinates']
        poly = ee.Geometry.Polygon(coords)
        w_preview.disabled = True
        w_export.disabled = True
        
dc = DrawControl(polyline={},circle={})
dc.on_draw(handle_draw)

def GetTileLayerUrl(ee_image_object):
    map_id = ee.Image(ee_image_object).getMapId()
    tile_url_template = "https://earthengine.googleapis.com/map/{mapid}/{{z}}/{{x}}/{{y}}?token={token}"
    return tile_url_template.format(**map_id)

w_text = widgets.Textarea(
    layout = widgets.Layout(width='75%'),
    value = 'Algorithm output',
    rows = 4,
    disabled = False
)
w_platform = widgets.RadioButtons(
    options=['SENTINEL/S2(VNIR/SWIR)','SENTINEL/S2(NIR/SWIR)','LANDSAT LC08','LANDSAT LE07','LANDSAT LT05'],
    value='SENTINEL/S2(VNIR/SWIR)',
    description='Platform:',
    disabled=False
)
w_startdate1 = widgets.Text(
    value='2018-04-01',
    placeholder=' ',
    description='Start T1:',
    disabled=False
)
w_enddate1 = widgets.Text(
    value='2018-05-01',
    placeholder=' ',
    description='End T1:',
    disabled=False
)
w_startdate2 = widgets.Text(
    value='2018-09-01',
    placeholder=' ',
    description='Start T2:',
    disabled=False
)
w_enddate2 = widgets.Text(
    value='2018-10-01',
    placeholder=' ',
    description='End T2:',
    disabled=False
)
w_exportname = widgets.Text(
    value='users/<username>/<path>',
    placeholder=' ',
    disabled=False
)

w_run = widgets.Button(description="Run")
w_preview = widgets.Button(description="Preview",disabled=True)
w_export = widgets.Button(description='Export to assets',disabled=True)
w_dates1 = widgets.VBox([w_startdate1,w_enddate1])
w_dates2 = widgets.VBox([w_startdate2,w_enddate2])
w_dates = widgets.HBox([w_platform,w_dates1,w_dates2])
w_exp = widgets.HBox([w_export,w_exportname])
w_go = widgets.HBox([w_run,w_preview,w_exp])
box = widgets.VBox([w_text,w_dates,w_go])

def on_widget_change(b):
    w_preview.disabled = True
    w_export.disabled = True

w_platform.observe(on_widget_change,names='value')
w_startdate1.observe(on_widget_change,names='value')
w_enddate1.observe(on_widget_change,names='value')
w_startdate2.observe(on_widget_change,names='value')
w_enddate2.observe(on_widget_change,names='value')

def on_run_button_clicked(b):
    global result,m,collection,count, \
           w_startdate1,w_enddate1,w_startdate2, \
           w_platfform,w_enddate2,w_changemap, \
           scale, \
           image1,image2, \
           madnames,coords,timestamp1,timestamp2
    try:
        w_text.value = 'running...'       
        coords = ee.List(poly.bounds().coordinates().get(0))
        
        cloudcover = 'CLOUD_COVER'
        scale = 30.0
        rgb = ['B4','B5','B7']
        if w_platform.value=='SENTINEL/S2(VNIR/SWIR)':
            collectionid = 'COPERNICUS/S2'
            scale = 10.0
            bands = ['B2','B3','B4','B8']
            rgb = ['B8','B4','B3']
            cloudcover = 'CLOUDY_PIXEL_PERCENTAGE'   
        elif w_platform.value=='SENTINEL/S2(NIR/SWIR)':
            collectionid = 'COPERNICUS/S2'
            scale = 20.0
            bands = ['B5','B6','B7','B8A','B11','B12']
            rgb = ['B5','B7','B11']
            cloudcover = 'CLOUDY_PIXEL_PERCENTAGE'    
        elif w_platform.value=='LANDSAT LC08':
            collectionid = 'LANDSAT/LC08/C01/T1_RT_TOA'
            bands = ['B2','B3','B4','B5','B6','B7']      
            rgb = ['B5','B6','B7']            
        elif w_platform.value=='LANDSAT LE07':
            collectionid  =  'LANDSAT/LE07/C01/T1_RT_TOA'
            bands = ['B1','B2','B3','B4','B5','B7']
        else:
            collectionid = 'LANDSAT/LT05/C01/T1_TOA'
            bands = ['B1','B2','B3','B4','B5','B7']                        
        collection1 = ee.ImageCollection(collectionid) \
                  .filterBounds(ee.Geometry.Point(coords.get(0))) \
                  .filterBounds(ee.Geometry.Point(coords.get(1))) \
                  .filterBounds(ee.Geometry.Point(coords.get(2))) \
                  .filterBounds(ee.Geometry.Point(coords.get(3))) \
                  .filterDate(ee.Date(w_startdate1.value), ee.Date(w_enddate1.value)) \
                  .sort(cloudcover, True) 
        count = collection1.size().getInfo()
        if count==0:
            raise ValueError('No images found for first time interval'+collectionid)               
        collection2 = ee.ImageCollection(collectionid) \
                  .filterBounds(ee.Geometry.Point(coords.get(0))) \
                  .filterBounds(ee.Geometry.Point(coords.get(1))) \
                  .filterBounds(ee.Geometry.Point(coords.get(2))) \
                  .filterBounds(ee.Geometry.Point(coords.get(3))) \
                  .filterDate(ee.Date(w_startdate2.value), ee.Date(w_enddate2.value)) \
                  .sort(cloudcover, True) 
        count = collection2.size().getInfo()
        if count==0:
            raise ValueError('No images found for second time interval')
        image1 = ee.Image(collection1.first()).select(bands)     
        timestamp1 = ee.Date(image1.get('system:time_start')).getInfo()
        timestamp1 = time.gmtime(int(timestamp1['value'])/1000)
        timestamp1 = time.strftime('%c', timestamp1)               
        systemid1 = image1.get('system:id').getInfo()
        cloudcover1 = image1.get(cloudcover).getInfo()
        image2 = ee.Image(collection2.first()).select(bands)     
        timestamp2 = ee.Date(image2.get('system:time_start')).getInfo()
        timestamp2 = time.gmtime(int(timestamp2['value'])/1000)
        timestamp2 = time.strftime('%c', timestamp2)               
        systemid2 = image2.get('system:id').getInfo()
        cloudcover2 = image2.get(cloudcover).getInfo()
        txt = 'Image1: %s \n'%systemid1
        txt += 'Acquisition date: %s, Cloud cover: %f \n'%(timestamp1,cloudcover1)
        txt += 'Image2: %s \n'%systemid2
        txt += 'Acquisition date: %s, Cloud cover: %f'%(timestamp2,cloudcover2)
        w_text.value = txt
        nbands = image1.bandNames().length()
        madnames = ['MAD'+str(i+1) for i in range(nbands.getInfo())]
#      co-register
        image2 = image2.register(image1,60)         
#      iMAD
        inputlist = ee.List.sequence(1,50)
        first = ee.Dictionary({'done':ee.Number(0),
                               'image':image1.addBands(image2).clip(poly),
                               'allrhos': [ee.List.sequence(1,nbands)],
                               'chi2':ee.Image.constant(0),
                               'MAD':ee.Image.constant(0)})         
        result = ee.Dictionary(inputlist.iterate(imad,first))                       
        w_preview.disabled = False
        w_export.disabled = False
#      display first image                
        if len(m.layers)>1:
            m.remove_layer(m.layers[1])
        rgb = image1.clip(poly).select(rgb).rename('r','g','b')
        ps = rgb.reduceRegion(ee.Reducer.percentile([2,98]),maxPixels=1e10).getInfo()
        mn = [ps['r_p2'],ps['g_p2'],ps['b_p2']]
        mx = [ps['r_p98'],ps['g_p98'],ps['b_p98']]
        m.add_layer(TileLayer(url=GetTileLayerUrl( rgb.visualize(min=mn,max=mx))))
    except Exception as e:
        w_text.value =  'Error: %s'%e

w_run.on_click(on_run_button_clicked)

def on_preview_button_clicked(b):
    global cmap
    try: 
        w_text.value = 'iteration started, please wait ...'
        MAD = ee.Image(result.get('MAD')).rename(madnames)
#      threshold        
        nbands = MAD.bandNames().length()
        chi2 = ee.Image(result.get('chi2')).rename(['chi2'])             
        pval = chi2cdf(chi2,nbands).subtract(1).multiply(-1)
        tst = pval.gt(ee.Image.constant(0.0001))
        MAD = MAD.where(tst,ee.Image.constant(0))              
        allrhos = ee.Array(result.get('allrhos')).toList()     
        txt = 'Canonical correlations: %s \n'%str(allrhos.get(-1).getInfo())
        w_text.value = txt
        if len(m.layers)>1:
            m.remove_layer(m.layers[1])      
        MAD1 = MAD.select(0).rename('b')
        ps = MAD1.reduceRegion(ee.Reducer.percentile([2,98])).getInfo()
        mn = ps['b_p2']
        mx = ps['b_p98']       
        m.add_layer(TileLayer(url=GetTileLayerUrl( MAD1.visualize(min=mn,max=mx))))
    except Exception as e:
        w_text.value =  'Error: %s\n Retry run/preview or export to assets'%e
    
w_preview.on_click(on_preview_button_clicked)   

def on_export_button_clicked(b):
    global w_exportname        
    try:
        MAD = ee.Image(result.get('MAD')).rename(madnames)
#      threshold        
        nbands = MAD.bandNames().length()
        chi2 = ee.Image(result.get('chi2')).rename(['chi2'])         
        pval = chi2cdf(chi2,nbands).subtract(1).multiply(-1)
        tst = pval.gt(ee.Image.constant(0.0001))
        MAD = MAD.where(tst,ee.Image.constant(0)) 
        allrhos = ee.Array(result.get('allrhos')).toList().slice(1,-1)          
#      radcal           
        ncmask = chi2cdf(chi2,nbands).lt(ee.Image.constant(0.05)).rename(['invarpix'])                     
        inputlist1 = ee.List.sequence(0,nbands.subtract(1))
        first = ee.Dictionary({'image':image1.addBands(image2),
                               'ncmask':ncmask,
                               'nbands':nbands,
                               'rect':poly,
                               'coeffs': ee.List([]),
                               'normalized':ee.Image()})
        result1 = ee.Dictionary(inputlist1.iterate(radcal,first))          
        coeffs = ee.List(result1.get('coeffs'))                    
        sel = ee.List.sequence(1,nbands)
        normalized = ee.Image(result1.get ('normalized')).select(sel)                                             
        MADs = ee.Image.cat(MAD,chi2,ncmask,image1.clip(poly),image2.clip(poly),normalized)        
        assexport = ee.batch.Export.image.toAsset(MADs,
                                    description='assetExportTask', 
                                    assetId=w_exportname.value,scale=scale,maxPixels=1e9)
        assexportid = str(assexport.id)
        w_text.value= 'Exporting change map to %s\n task id: %s'%(w_exportname.value,assexportid)
        assexport.start()  
    except Exception as e:
        w_text.value =  'Error: %s'%e        
#  export metadata to drive
    ninvar = ee.String(ncmask.reduceRegion(ee.Reducer.sum().unweighted(),
                                           scale=scale,maxPixels= 1e9).toArray().project([0]))
    metadata = ee.List(['IR-MAD: '+time.asctime(),  
                        'Platform: '+w_platform.value,
                        'Asset export name: '+w_exportname.value,   
                        'Timestamps: %s  %s'%(timestamp1,timestamp2)]) \
                        .cat(['Canonical Correlations:']) \
                        .cat(allrhos)  \
                        .cat(['Radiometric Normalization, Invariant Pixels:']) \
                        .cat([ninvar]) \
                        .cat(['Slope, Intercept, R:']) \
                        .cat(coeffs)  
    fileNamePrefix=w_exportname.value.replace('/','-')  
    gdexport = ee.batch.Export.table.toDrive(ee.FeatureCollection(metadata.map(makefeature)).merge(ee.Feature(poly)),
                         description='driveExportTask_meta', 
                         folder = 'EarthEngineImages',
                         fileNamePrefix=fileNamePrefix )
    w_text.value += '\n Exporting metadata to Drive/EarthEngineImages/%s\n task id: %s'%(fileNamePrefix,str(gdexport.id))            
    gdexport.start()                         
    
w_export.on_click(on_export_button_clicked) 

def run():
    global m,dc,center
    center = list(reversed(poly.centroid().coordinates().getInfo()))
    m = Map(center=center, zoom=10, layout={'height':'400px'})
    osm = basemap_to_tiles(basemaps.OpenStreetMap.HOT)
    mb = TileLayer(url="https://api.mapbox.com/styles/v1/mapbox/satellite-streets-v9/tiles/256/{z}/{x}/{y}?access_token=pk.eyJ1IjoibWNhbnR5IiwiYSI6ImNpcjRsMmJxazAwM3hoeW05aDA1cmNkNzMifQ.d2UbIugbQFk2lnU8uHwCsQ")
    sm_control = SplitMapControl(left_layer=osm,right_layer=mb)
    m.add_control(dc)
    m.add_control(sm_control)
    display(m)
    display(box)
    