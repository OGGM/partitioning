
import salem
from oggm import workflow,cfg,graphics,tasks
import geopandas as gpd
import matplotlib.pyplot as plt
import pickle
import os
import shutil
from partitioning import dividing_glaciers
from oggm.workflow import execute_entity_task

if __name__ == '__main__':
    cfg.initialize()
    base_dir = '/home/juliaeis/Dokumente/OGGM/work_dir/Central_Europe_all'
    cfg.PATHS['working_dir'] = os.path.join(base_dir, 'no_partitioning')
    cfg.PATHS['topo_dir'] = '/home/juliaeis/Dokumente/OGGM/input_data/topo'
    cfg.PARAMS['divides_gdf'] = gpd.GeoDataFrame()
    cfg.PARAMS['use_multiprocessing'] = False
    RGI_FILE = os.path.join(base_dir, '11_rgi50_CentralEurope.shp')

    # set dem to 40 meters
    cfg.PARAMS['d1'] = 40
    cfg.PARAMS['dmax'] = 40

    cfg.PARAMS['border'] = 10

    RUN_DIVIDES = False

    no_topo = ['RGI50-11.03813', 'RGI50-11.03814', 'RGI50-11.03815',
               'RGI50-11.03816', 'RGI50-11.03817', 'RGI50-11.03818',
               'RGI50-11.03819', 'RGI50-11.03820', 'RGI50-11.03821',
               'RGI50-11.03822', 'RGI50-11.03823', 'RGI50-11.03824',
               'RGI50-11.03825', 'RGI50-11.03826', 'RGI50-11.03827',
               'RGI50-11.03828', 'RGI50-11.03829', 'RGI50-11.03830',
               'RGI50-11.03831', 'RGI50-11.03832', 'RGI50-11.03833',
               'RGI50-11.03834', 'RGI50-11.03835', 'RGI50-11.03836']
    failed_topo = ['RGI50-11.00548', 'RGI50-11.03396']
    rgidf = salem.read_shapefile(RGI_FILE, cached=True)
    ID_s = pickle.load(open(os.path.join(base_dir, 'divided.pkl')))

    indices = [((i in ID_s) and (i not in failed_topo)) for i in rgidf.RGIId]
    #indices = [((i in ID_s)) for i in rgidf.RGIId]
    gdirs_orig = workflow.init_glacier_regions(rgidf[indices], reset=True)

    cfg.PATHS['working_dir'] = base_dir
    gdirs = workflow.init_glacier_regions(rgidf[indices], reset=False)

    task_list = [
        tasks.glacier_masks,
        tasks.compute_centerlines]
    for task in task_list:
        execute_entity_task(task, gdirs_orig)
        execute_entity_task(task, gdirs)

    for i, gdir_orig in enumerate(gdirs_orig):
        fig = plt.figure(figsize=(20, 10))
        ax0 = fig.add_subplot(1, 2, 2)
        ax1 = fig.add_subplot(1, 2, 1)
        graphics.plot_centerlines(gdir_orig, ax=ax1)
        graphics.plot_centerlines(gdirs[i], ax=ax0)
        plt.savefig(os.path.join(base_dir, 'plots', str(gdir_orig.rgi_id) + '.png'))
        #plt.show()
        plt.close()
