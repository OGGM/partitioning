import os

from oggm import cfg, workflow
from partitioning.postprocessing_py3 import postprocessing
import salem
import time
import multiprocessing as mp
import zipfile
import warnings
import copy

warnings.simplefilter(action='ignore', category=FutureWarning)


def run_div(gdir):
    input_shp = gdir.get_filepath('outlines')
    input_dem = gdir.get_filepath('dem')

    script = os.path.join(project, 'partitioning/examples/run_divides.py')
    os.system(python + ' ' + script + ' ' + input_shp + ' ' + input_dem)

if __name__ == '__main__':

    cfg.initialize()
    cfg.PARAMS['continue_on_error'] = True
    cfg.PARAMS['use_multiprocessing'] = True
    # set dem resolution to 40 meters
    cfg.PARAMS['grid_dx_method'] = 'fixed'
    cfg.PARAMS['fixed_dx'] = 40
    cfg.PARAMS['use_intersects'] = False
    cfg.PARAMS['border'] = 10
    cfg.PATHS['working_dir'] = os.environ.get('S_WORKDIR')

    rgi_region = str(os.environ.get('REGION')).zfill(2)
    rgi_file = os.environ.get('RGI_DATA')
    dir_name = [d for d in os.listdir(rgi_file) if
                d.startswith(rgi_region) and d.endswith(".zip")]
    zip_ref = zipfile.ZipFile(os.path.join(rgi_file, dir_name[0]), 'r')
    zip_ref.extractall(cfg.PATHS['working_dir'])
    shp_name = [name for name in zip_ref.namelist() if name.endswith(".shp")][
        0]
    zip_ref.close()
    rgi_file = os.path.join(cfg.PATHS['working_dir'], shp_name)

    rgidf = salem.read_shapefile(rgi_file, cached=True)
    rgidf['remarks'] = ""
    gdirs = workflow.init_glacier_regions(rgidf, reset=False)

    # path to python 2.7
    python = '/home/users/julia/python2_env/bin/python'
    # path where partioning is located
    project = '/home/users/julia'
    script = os.path.join(project, 'partitioning/examples/run_divides.py')

    start1 = time.time()
    pool = mp.Pool()
    pool.map(run_div, gdirs)
    pool.close()
    pool.join()

    filter_option = ['no', 'alt', 'all']

    filter_option = ['no', 'alt', 'all']
    for filter in filter_option:
        new_rgi = copy.deepcopy(rgidf)
        for gdir in gdirs:
            new_rgi = postprocessing(new_rgi, gdir, filter)

        sorted_rgi = new_rgi.sort_values('RGIId')
        sorted_rgi = sorted_rgi[['Area', 'OGGM_Area', 'Aspect', 'BgnDate',
                                 'CenLat', 'CenLon', 'Connect', 'EndDate',
                                 'Form', 'GLIMSId', 'Linkages', 'Lmax', 'Name',
                                 'O1Region', 'O2Region', 'RGIId', 'Slope',
                                 'Status', 'Surging', 'TermType', 'Zmax',
                                 'Zmed', 'Zmin', 'geometry', 'min_x', 'max_x',
                                 'min_y', 'max_y', 'remarks']]
        new_name = str(gdir.rgi_region) + '_dgi60_'+filter+'.shp'
        sorted_rgi.to_file(os.path.join(cfg.PATHS['working_dir'], new_name))
