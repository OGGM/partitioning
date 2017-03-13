from oggm import workflow, cfg
import salem
import geopandas as gpd
import shutil
import os
import pickle
from partitioning.core import dividing_glaciers

if __name__ == '__main__':
    cfg.initialize()
    base_dir = '/home/juliaeis/Dokumente/OGGM/work_dir/CentralEurope/2000-3000'
    cfg.PATHS['working_dir'] = base_dir
    cfg.PATHS['topo_dir'] = '/home/juliaeis/Dokumente/OGGM/input_data/topo'
    cfg.PARAMS['divides_gdf'] = gpd.GeoDataFrame()
    RGI_FILE = base_dir+'/outlines.shp'

    # set dem to 40 meters
    cfg.PARAMS['d1'] = 40
    cfg.PARAMS['dmax'] = 40

    rgidf = salem.read_shapefile(RGI_FILE, cached=True)
    gdirs = workflow.init_glacier_regions(rgidf, reset=True)
    RGI = 'RGI50-11.00002'
    for gdir in gdirs:
        print gdir.rgi_id

        input_shp = gdir.get_filepath('outlines', div_id=0)
        input_dem = gdir.get_filepath('dem', div_id=0)
        for fol in os.listdir(gdir.dir):
            if fol.startswith('divide'):
                shutil.rmtree(os.path.join(gdir.dir, fol))
        os.makedirs(os.path.join(gdir.dir, 'divide_01'))
        to_copy = [input_shp, os.path.join(gdir.dir, 'outlines.shx'),
                   os.path.join(gdir.dir, 'outlines.dbf')]
        for file in to_copy:
            shutil.copy(file, os.path.join(gdir.dir, 'divide_01'))
        # if gdir.rgi_id == RGI:
        n = dividing_glaciers(input_dem, input_shp)
        print gdir.rgi_id, 'no glaciers:', n
        '''
        # only check tidewaterglaciers
        n = []
        for gdir in gdirs:
            if not gdir.is_tidewater and 10 < gdir.rgi_area_km2 < 15:
                print (gdir.rgi_id, gdir.terminus_type, gdir.rgi_area_km2)
                n.append(gdir.rgi_id)
        pickle.dump(n, open(str(base_dir + "\\10-15.pkl"), "wb"), protocol=2)
        print (len(n))
        '''