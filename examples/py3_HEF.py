import os

from oggm import cfg,  workflow
from oggm.utils import get_demo_file
import matplotlib.pyplot as plt
import geopandas as gpd

if __name__ == '__main__':

    cfg.initialize()

    # Set Paths for OGGM
    cfg.PATHS['working_dir'] = 'path to working directory'


    # set dem resolution to 40 meters
    cfg.PARAMS['grid_dx_method'] = 'fixed'
    cfg.PARAMS['fixed_dx'] = 40
    cfg.PARAMS['border'] = 10
    cfg.PARAMS['use_intersects'] = False
    cfg.PARAMS['use_multiprocessing'] = False

    # get example shapefile initialize the model
    entity = gpd.read_file(get_demo_file('Hintereisferner_RGI6.shp'))
    hef = workflow.init_glacier_regions(entity, reset=False)[0]

    input_shp = hef.get_filepath('outlines')
    input_dem = hef.get_filepath('dem')

    # set paths to python 2.7 and to the partitioning package
    python = 'path to python 2.7'
    project = 'path to the partitioning package'

    # run_divides with python2.7
    script = os.path.join(project, 'examples/run_divides.py')
    os.system(python + ' ' + script + ' ' + input_shp + ' ' + input_dem)

    divides = gpd.read_file(os.path.join(hef.dir,'divides.shp'))

    print('Hintereisferner was divided in '+ str(len(divides)) + ' parts')

    #plot outlines of the divides
    divides.plot()
    plt.show()

