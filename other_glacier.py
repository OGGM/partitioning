import os
import geopandas as gpd
import numpy as np
from salem.utils import get_demo_file
import oggm
from oggm import tasks
import oggm.cfg as cfg
import oggm

from oggm import workflow
from oggm import tasks
from oggm.workflow import execute_entity_task
from oggm import graphics, utils
from oggm.utils import tuple2int

import salem

# Log message format
import logging
logging.basicConfig(format='%(asctime)s: %(name)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

# Module logger
log = logging.getLogger(__name__)

import matplotlib.pyplot as plt

import math
import fiona
import shutil
from shapely.geometry import mapping, shape
import rasterio
from partitioning import dividing_glaciers
import shutil
from rasterio.tools.mask import mask
from pygeoprocessing import routing

from skimage import img_as_float
from skimage.feature import peak_local_max
from shapely.geometry import Point,Polygon,MultiPolygon

if __name__ == '__main__':
    RGI_FILE ='/home/juliaeis/Dokumente/OGGM/work_dir/CentralEurope3000m+/outlines.shp'
    base_dir = os.path.join(os.path.expanduser('/home/juliaeis/Dokumente/OGGM/work_dir/CentralEurope3000m+'))
    import time
    start0=time.time()
    cfg.initialize()

    # Read RGI file
    rgidf = salem.read_shapefile(RGI_FILE, cached=True)
    f = open(base_dir + '/failed_glacier_id.txt', 'w')
    g = open(base_dir + '/succes_glacier_id.txt', 'w')
    for i in range(len(rgidf)):
        #if rgidf.iloc[i].RGIId not in ['RGI50-11.00116','RGI50-11.00181']:
        gdir=oggm.GlacierDirectory(rgidf.iloc[i], base_dir=base_dir)
        print gdir.dir
        if gdir.has_file('outlines', div_id=0) and gdir.has_file('dem', div_id=0):
            #tasks.define_glacier_region(gdir, entity=rgidf.iloc[i])
            pass
        else:
            tasks.define_glacier_region(gdir, entity=rgidf.iloc[i])
        input_shp = gdir.get_filepath('outlines', div_id=0)
        input_dem = gdir.get_filepath('dem', div_id=0)

        if os.path.isfile(gdir.dir+'/divide_01/outlines.shp'):
            os.remove(gdir.dir+'/divide_01/outlines.shp')
        try:
            no=dividing_glaciers(input_dem, input_shp)
            if no is not 1:
                g.write(str(rgidf.iloc[i].RGIId) + '/n')
        except:
            f.write(str(rgidf.iloc[i].RGIId)+'/n' )
    f.closed
    g.closed

    '''
    base_dir = os.path.join(os.path.expanduser('/home/juliaeis/Dokumente/OGGM/work_dir'), 'Scandinavia')
    #entity = gpd.GeoDataFrame.from_file(get_demo_file('Hintereisferner.shp')).iloc[0]
    #gdir = oggm.GlacierDirectory(entity, base_dir=base_dir)
    for dir in os.listdir(base_dir):
        if dir.startswith('RGI50'):
            gdir=oggm.GlacierDirectory(dir,base_dir=base_dir)
            entity = gpd.GeoDataFrame.from_file(gdir.dir+'/outlines.shp').iloc[0]
            print gdir.dir
            #check if required files exists
            if gdir.has_file('outlines',div_id=0) and gdir.has_file('dem', div_id=0):
                pass
            else:
                tasks.define_glacier_region(gdir,entity=entity)
            #tasks.glacier_masks(gdir)
            ###################preprocessing########################
            input_shp =gdir.get_filepath('outlines',div_id=0)
            input_dem=gdir.get_filepath('dem',div_id=0)

            #get pixel size
            with rasterio.open(input_dem) as dem:
                global pixelsize
                pixelsize=int(dem.transform[1])
            dividing_glaciers(input_dem, input_shp)


    print time.time()-start0

    #test if it works

    from oggm import graphics

    tasks.glacier_masks(gdir)
    tasks.compute_centerlines(gdir)
    tasks.compute_downstream_lines(gdir)
    tasks.catchment_area(gdir)
    tasks.initialize_flowlines(gdir)
    tasks.catchment_width_geom(gdir)

    graphics.plot_centerlines(gdir)
    plt.show()

'''