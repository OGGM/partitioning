from oggm import workflow, cfg
import salem
import geopandas as gpd
import shutil
import os
import pickle
from partitioning.core import dividing_glaciers
import pickle

if __name__ == '__main__':
    cfg.initialize()
    base_dir = '/home/juliaeis/Dokumente/OGGM/work_dir/Central_Europe_all'
    cfg.PATHS['working_dir'] = base_dir
    cfg.PATHS['topo_dir'] = '/home/juliaeis/Dokumente/OGGM/input_data/topo'
    cfg.PARAMS['divides_gdf'] = gpd.GeoDataFrame()
    cfg.PARAMS['use_multiprocessing'] = False
    RGI_FILE = os.path.join(base_dir, '11_rgi50_CentralEurope.shp')

    # set dem to 40 meters
    cfg.PARAMS['d1'] = 40
    cfg.PARAMS['dmax'] = 40
    cfg.PARAMS['border'] = 10

    rgidf = salem.read_shapefile(RGI_FILE, cached=True)

    # no topo in alps
    no_topo = ['RGI50-11.03813', 'RGI50-11.03814', 'RGI50-11.03815',
               'RGI50-11.03816', 'RGI50-11.03817', 'RGI50-11.03818',
               'RGI50-11.03819', 'RGI50-11.03820', 'RGI50-11.03821',
               'RGI50-11.03822', 'RGI50-11.03823', 'RGI50-11.03824',
               'RGI50-11.03825', 'RGI50-11.03826', 'RGI50-11.03827',
               'RGI50-11.03828', 'RGI50-11.03829', 'RGI50-11.03830',
               'RGI50-11.03831', 'RGI50-11.03832', 'RGI50-11.03833',
               'RGI50-11.03834', 'RGI50-11.03835', 'RGI50-11.03836']

    ID_s = pickle.load(open(os.path.join(base_dir, 'divided.pkl')))
    indices = [(i not in no_topo) and (i in ID_s) for i in rgidf.RGIId]
    gdirs = workflow.init_glacier_regions(rgidf[indices], reset=True)

    failed = []
    for gdir in gdirs:
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

        n = dividing_glaciers(input_dem, input_shp)
        print gdir.rgi_id, 'no glaciers:', n


