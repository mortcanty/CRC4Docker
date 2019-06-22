'''
Created on 21.06.2018

@author: mort

ipywidget interface to the GEE for sequential SAR change detection

'''
import ee, time, warnings, math
import ipywidgets as widgets
from IPython.display import display
from ipyleaflet import (Map,DrawControl,TileLayer,
                        basemaps,basemap_to_tiles,
                        LayersControl,
                        SplitMapControl)
from auxil.eeWishart import omnibus

ee.Initialize()

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

poly = ee.Geometry.Polygon([[6.30154, 50.948329], [6.293307, 50.877329], 
                            [6.427091, 50.875595], [6.417486, 50.947464], 
                            [6.30154, 50.948329]])

center = list(reversed(poly.centroid().coordinates().getInfo()))

def get_incidence_angle(image):
    ''' grab the mean incidence angle '''
    result = ee.Image(image).select('angle') \
           .reduceRegion(ee.Reducer.mean(),geometry=poly,maxPixels=1e9) \
           .get('angle') \
           .getInfo()
    if result is not None:
        return round(result,2)
    else:
#      incomplete overlap, so use all of the image geometry        
        return round(ee.Image(image).select('angle') \
           .reduceRegion(ee.Reducer.mean(),maxPixels=1e9) \
           .get('angle') \
           .getInfo(),2)

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
    enl = ee.Number(ee.Dictionary(prev).get('enl'))   
    imlist = imlist.add(ee.Image(current).multiply(enl).clip(poly))
    return ee.Dictionary({'imlist':imlist,'poly':poly,'enl':enl})

def makefeature(data):
    ''' for exporting as CSV to Drive '''
    return ee.Feature(None, {'data': data})

def handle_draw(self, action, geo_json):
    global poly
    if action == 'created':
        coords =  geo_json['geometry']['coordinates']
        poly = ee.Geometry.Polygon(coords)
        w_preview.disabled = True
        w_export_ass.disabled = True
        
dc = DrawControl(polyline={},circlemarker={})
dc.rectangle = {"shapeOptions": {"fillColor": "#0000ff","color": "#0000ff","fillOpacity": 0.1}}
dc.on_draw(handle_draw)

def GetTileLayerUrl(ee_image_object):
    map_id = ee.Image(ee_image_object).getMapId()
    tile_url_template = "https://earthengine.googleapis.com/map/{mapid}/{{z}}/{{x}}/{{y}}?token={token}"
    return tile_url_template.format(**map_id)

w_collection = widgets.Text(
    value='COPERNICUS/S1_GRD',
#    value='COPERNICUS/S1_GRD',
    placeholder=' ',
    description='Collection:',
    disabled=False
)
w_enl = widgets.BoundedFloatText(
    value=4.4,
    min=3.0,
    max=20.0,
    step=0.1,
    description='ENL:',
    disabled=False
)
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
w_platform = widgets.RadioButtons(
    options=['Both','A','B'],
     value='Both',
    description='Platform:',
    disabled=False
)
w_relativeorbitnumber = widgets.IntText(
    value=0,
    description='Rel orbit:',
    disabled=False
)
w_exportname = widgets.Text(
    value='users/<username>/<path>',
    placeholder=' ',
    disabled=False
)
w_exportscale = widgets.Text(
    value='10',
    placeholder=' ',
    description='at scale ',
    disabled=False
)
w_startdate = widgets.Text(
    value='2018-04-01',
    placeholder=' ',
    description='Start date:',
    disabled=False
)
w_enddate = widgets.Text(
    value='2018-10-01',
    placeholder=' ',
    description='End date:',
    disabled=False
)
w_median = widgets.Checkbox(
    value=False,
    description='3x3 Median filter',
    disabled=False
)
w_Q = widgets.Checkbox(
    value=True,
    description='Use Q',
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
w_text = widgets.Textarea(
    layout = widgets.Layout(width='100%'),
    value = 'Algorithm output',
    rows = 4,
    disabled = False
)

w_collect = widgets.HBox([w_collection,w_enl])
w_run = widgets.Button(description="Run")
w_preview = widgets.Button(description="Preview",disabled=True)
w_export_ass = widgets.Button(description='Export to asset==>',disabled=True)
w_dates = widgets.HBox([w_relativeorbitnumber,w_startdate,w_enddate])
w_orbit = widgets.HBox([w_orbitpass,w_platform,w_changemap,w_opacity])
w_exp_ass = widgets.HBox([w_export_ass,w_exportname,w_exportscale])
w_signif = widgets.HBox([w_significance,w_median,w_Q])
w_rse = widgets.HBox([w_run,w_preview,w_exp_ass])

box = widgets.VBox([w_text,w_collect,w_dates,w_orbit,w_signif,w_rse])

def on_widget_change(b):
    w_preview.disabled = True
    w_export_ass.disabled = True

w_orbitpass.observe(on_widget_change,names='value')
w_platform.observe(on_widget_change,names='value')
w_relativeorbitnumber.observe(on_widget_change,names='value')
w_startdate.observe(on_widget_change,names='value')
w_collection.observe(on_widget_change,names='value')
w_enl.observe(on_widget_change,names='value')
w_enddate.observe(on_widget_change,names='value')
w_median.observe(on_widget_change,names='value')
w_Q.observe(on_widget_change,names='value')
w_significance.observe(on_widget_change,names='value')

def on_run_button_clicked(b):
    global result,m,collection,count,timestamplist1, \
           w_startdate,w_enddate,w_orbitpass,w_changemap, \
           w_relativeorbitnumber,w_significance,w_median,w_Q, \
           mean_incidence,collectionmean,coords
    try:       
        if w_collection.value == 'COPERNICUS/S1_GRD':    
            w_text.value = 'running on GEE archive ...' 
            coords = ee.List(poly.bounds().coordinates().get(0))
            collection = ee.ImageCollection(w_collection.value) \
                      .filterBounds(ee.Geometry.Point(coords.get(0))) \
                      .filterBounds(ee.Geometry.Point(coords.get(1))) \
                      .filterBounds(ee.Geometry.Point(coords.get(2))) \
                      .filterBounds(ee.Geometry.Point(coords.get(3))) \
                      .filterDate(ee.Date(w_startdate.value), ee.Date(w_enddate.value)) \
                      .filter(ee.Filter.eq('transmitterReceiverPolarisation', ['VV','VH'])) \
                      .filter(ee.Filter.eq('resolution_meters', 10)) \
                      .filter(ee.Filter.eq('instrumentMode', 'IW')) \
                      .filter(ee.Filter.eq('orbitProperties_pass', w_orbitpass.value))   
            if w_relativeorbitnumber.value > 0:
                collection = collection.filter(ee.Filter.eq('relativeOrbitNumber_start', int(w_relativeorbitnumber.value)))   
            if w_platform.value != 'Both':
                collection = collection.filter(ee.Filter.eq('platform_number', w_platform.value))         
            collection = collection.sort('system:time_start') 
            acquisition_times = ee.List(collection.aggregate_array('system:time_start')).getInfo()
            count = len(acquisition_times) 
            if count<2:
                raise ValueError('Less than 2 images found')
            timestamplist = []
            for timestamp in acquisition_times:
                tmp = time.gmtime(int(timestamp)/1000)
                timestamplist.append(time.strftime('%x', tmp))  
    #      make timestamps in YYYYMMDD format            
            timestamplist = [x.replace('/','') for x in timestamplist]  
            timestamplist = ['T20'+x[4:]+x[0:4] for x in timestamplist]
    #      in case of duplicates add running integer
            timestamplist1 = [timestamplist[i] + '_' + str(i+1) for i in range(len(timestamplist))]    
            relativeorbitnumbers = map(int,ee.List(collection.aggregate_array('relativeOrbitNumber_start')).getInfo())
            rons = list(set(relativeorbitnumbers))
            txt = 'running on GEE archive ...'
            txt += 'Images found: %i, platform: %s \n'%(count,w_platform.value)
            txt += 'Acquisition dates: %s\n'%str(timestamplist)
            txt += 'Relative orbit numbers: '+str(rons)+'\n'
            if len(rons)==1:
                mean_incidence = get_incidence_angle(collection.first())
                txt += 'Mean incidence angle: %f'%mean_incidence
            else:
                txt += 'Mean incidence angle: (select one rel. orbit)'
            w_text.value = txt
            pcollection = collection.map(get_vvvh)
            pList = pcollection.toList(100)   
            first = ee.Dictionary({'imlist':ee.List([]),'poly':poly,'enl':ee.Number(w_enl.value)}) 
            imList = ee.Dictionary(pList.iterate(clipList,first)).get('imlist')
            collectionmean = collection.mean().select(0).clip(poly)
            collmeanviz = collectionmean.visualize(min=-15, max=5, opacity=w_opacity.value) 
        else:
            txt = 'running on local collection ...\n' 
            collection = ee.ImageCollection(w_collection.value)
            count = collection.size().getInfo()            
            w_exportscale.value = str(collection.first().projection().nominalScale().getInfo())           
            txt += 'Images found: %i'%count
            timestamplist1 = ['T%i'%(i+1) for i in range(count)]
            w_text.value = txt
            imList = collection.toList(100)
            collectionmean = collection.mean().select(0)
            collmeanviz = collectionmean.log().visualize(min=5, max=15, opacity=w_opacity.value)
#      run the algorithm        
        result = ee.Dictionary(omnibus(imList,w_significance.value,w_enl.value,w_median.value,w_Q.value))
        w_preview.disabled = False
#      display mean of the full collection                
        if len(m.layers)>1:
            m.remove_layer(m.layers[1])
        m.add_layer(TileLayer(url=GetTileLayerUrl(collmeanviz)))
    except Exception as e:
        w_text.value =  'Error: %s'%e

w_run.on_click(on_run_button_clicked)

def on_preview_button_clicked(b):
    global result,m,cmap,smap,fmap,bmap,w_changemap
    jet = 'black,blue,cyan,yellow,red'
    smap = ee.Image(result.get('smap')).byte()
    cmap = ee.Image(result.get('cmap')).byte()
    fmap = ee.Image(result.get('fmap')).byte() 
    bmap = ee.Image(result.get('bmap')).byte()
    
    cmaps = ee.Image.cat(cmap,smap,fmap,bmap).rename(['cmap','smap','fmap']+timestamplist1[1:])
    downloadpath = cmaps.getDownloadUrl({'scale':10})
    w_text.value = 'Download change maps from this URL:\n'+downloadpath+'\nNote: This may be unreliable'
    
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
    m.add_layer(TileLayer(url=GetTileLayerUrl( mp.visualize(min=0, max=mx, palette=jet,opacity = w_opacity.value))))
    w_export_ass.disabled = False
    
w_preview.on_click(on_preview_button_clicked)   

def on_export_ass_button_clicked(b):
    global w_exportname
    background = collectionmean.add(15).divide(15)
    if w_collection.value == 'COPERNICUS/S1_GRD':  
        collection1 = ee.ImageCollection('COPERNICUS/S2') \
                        .filterBounds(ee.Geometry.Point(coords.get(0))) \
                        .filterBounds(ee.Geometry.Point(coords.get(1))) \
                        .filterBounds(ee.Geometry.Point(coords.get(2))) \
                        .filterBounds(ee.Geometry.Point(coords.get(3))) \
                        .filterDate(ee.Date(w_startdate.value),ee.Date(w_enddate.value)) \
                        .sort('CLOUDY_PIXEL_PERCENTAGE',True) \
                        .filterMetadata('CLOUDY_PIXEL_PERCENTAGE','less_than',1.0) 
        count1 = collection1.size().getInfo()
        if count1>0:
    #      use sentinel-2 as video background if available                       
            background = ee.Image(collection1.first()) \
                                   .clip(poly) \
                                   .select('B8') \
                                   .divide(5000)    
    cmaps = ee.Image.cat(cmap,smap,fmap,bmap,background).rename(['cmap','smap','fmap']+timestamplist1[1:]+['background'])                
    assexport = ee.batch.Export.image.toAsset(cmaps,
                                description='assetExportTask', 
                                assetId=w_exportname.value,scale=w_exportscale.value,maxPixels=1e9)
    assexportid = str(assexport.id)
    w_text.value= 'Exporting change maps to %s\n task id: %s'%(w_exportname.value,assexportid)
    assexport.start()  
    
#  export metadata to drive
    if w_collection.value == 'COPERNICUS/S1_GRD': 
        times = [timestamp[1:9] for timestamp in timestamplist1]
        metadata = ee.List(['SEQUENTIAL OMNIBUS: '+time.asctime(),  
                            'Collection: '+w_collection.value,
                            'Asset export name: '+w_exportname.value,  
                            'ENL: '+str(w_enl.value),
                            'Export scale (m): '+w_exportscale.value,
                            'Nominal scale (m): '+str(cmap.projection().nominalScale().getInfo()),
                            'Orbit pass: '+w_orbitpass.value,    
                            'Significance: '+str(w_significance.value),  
                            'Series length: '+str(len(times)),
                            'Timestamps: '+str(times)[1:-1],
                            'Rel orbit numbers: '+str(w_relativeorbitnumber.value),
                            'Platform: '+w_platform.value,
                            'Mean incidence angles: '+str(mean_incidence),
                            'Used 3x3 median filter: '+str(w_median.value),
                            'Used Q test: '+str(w_Q.value)]) \
                            .cat(['Polygon:']) \
                            .cat(poly.getInfo()['coordinates'][0]) 
    else:
        metadata = ee.List(['SEQUENTIAL OMNIBUS: '+time.asctime(),  
                            'Collection: '+w_collection.value,
                            'Asset export name: '+w_exportname.value,  
                            'ENL: '+str(w_enl.value),  
                            'Export scale (m): '+w_exportscale.value,
                            'Nominal scale (m): '+str(cmap.projection().nominalScale().getInfo()),
                            'Significance: '+str(w_significance.value),  
                            'Series length: '+str(count),
                            'Used 3x3 median filter: '+str(w_median.value),
                            'Used Q test: '+str(w_Q.value)]) 
    fileNamePrefix=w_exportname.value.replace('/','-')  
    gdexport = ee.batch.Export.table.toDrive(ee.FeatureCollection(metadata.map(makefeature)),
                         description='driveExportTask_meta', 
                         folder = 'EarthEngineImages',
                         fileNamePrefix=fileNamePrefix )
    w_text.value += '\n Exporting metadata to Drive/EarthEngineImages/%s\n task id: %s'%(fileNamePrefix,str(gdexport.id))            
    gdexport.start()                         
    
w_export_ass.on_click(on_export_ass_button_clicked) 

def run():
    global m,dc,center
    center = list(reversed(poly.centroid().coordinates().getInfo()))
    m = Map(center=center, zoom=10, layout={'height':'400px'})
    osm = basemap_to_tiles(basemaps.OpenStreetMap.HOT)
    mb = TileLayer(url="https://api.mapbox.com/styles/v1/mapbox/satellite-streets-v9/tiles/256/{z}/{x}/{y}?access_token=pk.eyJ1IjoibWNhbnR5IiwiYSI6ImNpcjRsMmJxazAwM3hoeW05aDA1cmNkNzMifQ.d2UbIugbQFk2lnU8uHwCsQ")
    sm_control = SplitMapControl(left_layer=osm,right_layer=mb)
    m.add_control(dc)
    m.add_control(sm_control)
    m.add_control(LayersControl())
    display(m)
    display(box)
    