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
    RGI_FILE ='/home/juliaeis/Dokumente/OGGM/work_dir/Alaska/RGI50-01.10689/outlines.shp'
    base_dir = os.path.join(os.path.expanduser('/home/juliaeis/Dokumente/OGGM/work_dir/Alaska'))
    import time
    start0=time.time()
    cfg.initialize()

    cfg.PATHS['dem_file']= os.path.join('/home/juliaeis/Dokumente/OGGM/work_dir/Alaska/dem.tif')
    # Read RGI file
    rgidf = salem.read_shapefile(RGI_FILE, cached=True)

    for i in range(len(rgidf)):
        if rgidf.iloc[i].RGIId in ['RGI50-01.10689']:
            gdir=oggm.GlacierDirectory(rgidf.iloc[i], base_dir=base_dir)
            print gdir.dir
            #tasks.define_glacier_region(gdir, entity=rgidf.iloc[i])

            if gdir.has_file('outlines', div_id=0) and gdir.has_file('dem', div_id=0):
                #tasks.define_glacier_region(gdir, entity=rgidf.iloc[i])
                pass
            else:
                tasks.define_glacier_region(gdir, entity=rgidf.iloc[i])
            input_shp = gdir.get_filepath('outlines', div_id=0)
            input_dem = gdir.get_filepath('dem', div_id=0)

            if os.path.isfile(gdir.dir+'/divide_01/outlines.shp'):
                os.remove(gdir.dir+'/divide_01/outlines.shp')

            no,k=dividing_glaciers(input_dem, input_shp)
            '''
            tasks.glacier_masks(gdir)
            tasks.compute_centerlines(gdir)
            graphics.plot_centerlines(gdir)
            '''