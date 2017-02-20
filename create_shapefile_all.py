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

#from partitioning import dividing_glaciers

if __name__ == '__main__':

    cfg.initialize()
    base_dir = 'C:\\Users\\Julia\\OGGM_wd\\CentralEurope\\2000-3000'
    #base_dir='/home/juliaeis/Dokumente/OGGM/work_dir/Alaska/land-terminating'
    cfg.PATHS['working_dir'] = base_dir
    #cfg.PATHS['dem_file']='/home/juliaeis/Dokumente/OGGM/work_dir/Alaska/dem.tif'
    RGI_FILE=base_dir+'/outlines.shp'
    RUN_DIVIDES=False

    rgidf = salem.read_shapefile(RGI_FILE, cached=True)
    gdirs = workflow.init_glacier_regions(rgidf)

    with fiona.open(RGI_FILE,'r') as outlines:
        schema=outlines.schema
        schema['properties'].update({(u'min_x', 'str:80'), (u'max_x', 'str:80'), (u'min_y', 'str:80'), (u'max_y', 'str:80')})
        crs_new=outlines.crs

    p1=Proj(crs_new)
    destination = Proj(crs_new)
    with fiona.open(base_dir+ '/divides_2000-3000.shp', "w", "ESRI Shapefile",schema, crs_new) as all:
        for gdir in gdirs:
            if RUN_DIVIDES:
                #delete folders including divide_01
                for fol in os.listdir(gdir.dir):
                    if fol.startswith('divide'):
                        shutil.rmtree(gdir.dir + '/' + fol)
                os.makedirs(gdir.dir + '/divide_01')
                for file in [gdir.get_filepath('outlines',div_id=0), gdir.dir + '/outlines.shx', gdir.dir + '/outlines.dbf']:
                    shutil.copy(file, gdir.dir + '/divide_01')

                input2_dem=os.path.dirname(gdir.get_filepath('dem',div_id=0))+'/dem2.tif'
                os.system('gdalwarp -tr 40 40 -r cubicspline -overwrite ' + gdir.get_filepath('dem',div_id=0) + ' ' + input2_dem)
                #dividing_glaciers(input2_dem,gdir.get_filepath('outlines',div_id=0))


            if gdir.n_divides is not 1:
                print (gdir.rgi_id)
                for i in range(gdir.n_divides):
                    with fiona.open(gdir.get_filepath('outlines', div_id=i+1),'r') as result:
                        res=result.next()
                        g1=shape(res['geometry'])

                        project = partial(
                            pyproj.transform,
                            pyproj.Proj(result.crs),  # source coordinate system
                            pyproj.Proj(crs_new))      # destination coordinate system

                        g2 = transform(project, g1)  # apply projection
                        all.write({'properties': res['properties'], 'geometry': mapping(g2)})
    print ('done')
