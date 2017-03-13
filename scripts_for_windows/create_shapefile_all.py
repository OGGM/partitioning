import os
import shutil
from functools import partial
import fiona
from shapely.geometry import mapping, shape
from shapely.ops import transform
import pyproj
from pyproj import Proj

import salem
from oggm import workflow
import oggm.cfg as cfg
from partitioning import dividing_glaciers


def run_divides(gdir1):
    # delete folders including divide_01
    for fol in os.listdir(gdir1.dir):
        if fol.startswith('divide'):
            shutil.rmtree(gdir1.dir + '/' + fol)
    os.makedirs(gdir1.dir + '/divide_01')
    to_copy = [gdir1.get_filepath('outlines', div_id=0),
               gdir1.dir + '/outlines.shx', gdir1.dir + '/outlines.dbf']
    for file in to_copy:
        shutil.copy(file, gdir.dir + '/divide_01')
    dividing_glaciers(gdir.get_filepath('dem', div_id=0),
                      gdir.get_filepath('outlines', div_id=0))


if __name__ == '__main__':
    cfg.initialize()
    base_dir = 'C:\\Users\\Julia\\OGGM_wd\\Alaska_non_tidewater'
    cfg.PATHS['working_dir'] = base_dir

    RGI_FILE = base_dir + '\\01_rgi50_Alaska.shp'
    RUN_DIVIDES = False

    rgidf = salem.read_shapefile(RGI_FILE, cached=True)
    gdirs = workflow.init_glacier_regions(rgidf)

    with fiona.open(RGI_FILE, 'r') as outlines:
        schema = outlines.schema
        add = {(u'min_x', 'str:80'), (u'max_x', 'str:80'),
               (u'min_y', 'str:80'), (u'max_y', 'str:80')}
        schema['properties'].update(add)
        crs_new = outlines.crs

    p1 = Proj(crs_new)
    destination = Proj(crs_new)
    with fiona.open(base_dir + '/3000-4000.shp', "w", "ESRI Shapefile",
                    schema, crs_new) as all:
        for gdir in gdirs:
            if RUN_DIVIDES:
                run_divides(gdir)
            if gdir.n_divides is not 1:
                print (gdir.rgi_id)
                for i in range(gdir.n_divides):
                    shp = gdir.get_filepath('outlines', div_id=i+1)
                    with fiona.open(shp, 'r') as result:
                        res = result.next()
                        g1 = shape(res['geometry'])
                        project = partial(
                            pyproj.transform,
                            # source coordinate system
                            pyproj.Proj(result.crs),
                            # destination coordinate system
                            pyproj.Proj(crs_new))

                        g2 = transform(project, g1)  # apply projection
                        all.write({'properties': res['properties'],
                                   'geometry': mapping(g2)})
    print ('done')
