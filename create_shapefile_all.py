
import salem
from oggm import workflow
import oggm.cfg as cfg
import fiona
from shapely.geometry import mapping, shape,Polygon

from partitioning import dividing_glaciers
from functools import partial
import pyproj
from shapely.ops import transform

if __name__ == '__main__':
    import time
    from pyproj import Proj
    cfg.initialize()

    start0=time.time()
    base_dir = '/home/juliaeis/Dokumente/OGGM/work_dir/CentralEurope3000m+'
    cfg.PATHS['working_dir'] = base_dir
    RGI_FILE=base_dir+'/outlines.shp'
    rgidf = salem.read_shapefile(RGI_FILE, cached=True)
    gdirs = workflow.init_glacier_regions(rgidf)

    with fiona.open(RGI_FILE,'r') as outlines:
        schema=outlines.schema
        schema['properties'].update({(u'min_x', 'str:80'), (u'max_x', 'str:80'), (u'min_y', 'str:80'), (u'max_y', 'str:80')})
        crs_new=outlines.crs

    p1=Proj(crs_new)
    destination = Proj(crs_new)
    with fiona.open(base_dir+ '/all_divides.shp', "w", "ESRI Shapefile",schema, crs_new) as all:
        for gdir in gdirs:

            if gdir.n_divides is not 1:
                for i in range(gdir.n_divides):
                    with fiona.open(gdir.get_filepath('outlines', div_id=i+1),'r') as result:
                        res=result.next()
                        g1=shape(res['geometry'])

                        project = partial(
                            pyproj.transform,
                            pyproj.Proj(result.crs),  # source coordinate system
                            pyproj.Proj(crs_new))      # destination coordinate system

                        g2 = transform(project, g1)  # apply projection
                        all.write({'properties': res['properties'], 'geometry': mapping(Polygon(g2.exterior))})
