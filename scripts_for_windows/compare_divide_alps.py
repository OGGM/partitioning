import os

import salem
from oggm import tasks,workflow
import oggm.cfg as cfg

import fiona
import shutil
from shapely.geometry import mapping, shape

from partitioning import dividing_glaciers
from functools import partial
import pyproj
from shapely.ops import transform

if __name__ == '__main__':
    import time
    from pyproj import Proj
    cfg.initialize()

    start0=time.time()
    base_dir = '/home/juliaeis/Dokumente/OGGM/work_dir/divides'
    cfg.PATHS['working_dir'] = base_dir

    rgids=[]
    with fiona.open('/home/juliaeis/Dokumente/OGGM/work_dir/divides/manuel'+'/divide_alps.shp', 'r') as alps:
        crs_new=alps.crs
        for glac in alps:
            if glac['properties']['RGIId'] not in rgids:
                rgids.append(glac['properties']['RGIId'])
    RGI_FILE=base_dir+'/11_rgi50_CentralEurope.shp'


    rgidf = salem.read_shapefile(RGI_FILE, cached=True)
    id=[(i in rgids) for i in rgidf.RGIId]

    rgidf = rgidf.iloc[id]
    gdirs = workflow.init_glacier_regions(rgidf)

    pre_pro = [
        tasks.glacier_masks]

    for task in pre_pro:
        execute_entity_task(task, gdirs)
    with fiona.open(RGI_FILE,'r') as outlines:
        schema=outlines.schema
        schema['properties'].update({(u'min_x', 'str:80'), (u'max_x', 'str:80'), (u'min_y', 'str:80'), (u'max_y', 'str:80')})

    j=0
    crs_list=[]
    p1=Proj(crs_new)
    destination = Proj(crs_new)
    with fiona.open(base_dir+ '/all_divides.shp', "w", "ESRI Shapefile",schema, crs_new) as all:
        for gdir in gdirs:

            input_shp = gdir.get_filepath('outlines', div_id=0)
            input_dem = gdir.get_filepath('dem', div_id=0)
            for fol in os.listdir(gdir.dir):
                if fol.startswith('divide'):
                    shutil.rmtree(gdir.dir + '/' + fol)
            n, k = dividing_glaciers(input_dem, input_shp)
            if n is not 1:

                for i in range(n):

                    with fiona.open(gdir.get_filepath('outlines', div_id=i+1),'r') as result:
                        res=result.next()
                        g1=shape(res['geometry'])

                        project = partial(
                            pyproj.transform,
                            pyproj.Proj(result.crs),  # source coordinate system
                            pyproj.Proj(crs_new))  # destination coordinate system

                        g2 = transform(project, g1)  # apply projection
                        all.write({'properties': res['properties'], 'geometry': mapping(g2)})
                        '''
                        original=Proj(result.crs)
                        res=result.next()
                        print shape(res['geometry']).area
                        index=0
                        for point in res['geometry']['coordinates'][0]:

                            long, lat = point
                        #long, lat =res['geometry']['coordinates']
                            x, y = transform(original, destination, long, lat)
                            res['geometry']['coordinates'][0][index]=(x,y)
                            index=index+1
                        print Polygon(res['geometry']['coordinates'][0]).area
                        all.write({'properties': res['properties'], 'geometry': mapping(shape(res['geometry']))})
                        '''