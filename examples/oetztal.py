import os
import salem
from oggm import cfg, workflow, tasks, utils
from oggm.utils import get_demo_file
from partitioning.postprocessing_py3 import postprocessing

if __name__ == '__main__':
    cfg.initialize()
    # set Paths
    WORKING_DIR = 'path to working dir'
    python = 'path to python 2.7 executable'
    project = 'path to the partitioning project'

    # OGGM parameter to set
    utils.mkdir(WORKING_DIR, reset=False)
    cfg.PATHS['working_dir'] = WORKING_DIR
    cfg.PATHS['dem_file'] = get_demo_file('srtm_oetztal.tif')

    cfg.PARAMS['use_multiprocessing'] = True
    cfg.PARAMS['grid_dx_method'] = 'fixed'
    cfg.PARAMS['fixed_dx'] = 40
    cfg.PARAMS['border'] = 10
    cfg.PARAMS['use_intersects'] = False
    cfg.PARAMS['use_multiprocessing'] = False

    rgi = get_demo_file('rgi_oetztal.shp')
    rgidf = salem.read_shapefile(rgi)
    gdirs = workflow.init_glacier_regions(rgidf)
    workflow.execute_entity_task(tasks.glacier_masks, gdirs)

    script = os.path.join(project, 'examples/run_divides.py')

    for gdir in gdirs:
        input_shp = gdir.get_filepath('outlines')
        input_dem = gdir.get_filepath('dem')
        os.system(python + ' ' + script + ' ' + input_shp + ' ' + input_dem)
        rgidf = postprocessing(rgidf,gdir,'all')

    # saves the output to a shapefile
    rgidf.to_file(os.path.join(cfg.PATHS['working_dir'],'DividedInventory_Oetztal.shp'))