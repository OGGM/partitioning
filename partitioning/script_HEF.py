import os

from oggm import cfg, tasks, graphics, workflow
from oggm.utils import get_demo_file
import matplotlib.pyplot as plt
import geopandas as gpd

if __name__ == '__main__':

    cfg.initialize()
    cfg.set_divides_db()
    cfg.PARAMS['use_multiprocessing'] = False
    # set dem resolution to 40 meters
    cfg.PARAMS['grid_dx_method'] = 'fixed'
    cfg.PARAMS['fixed_dx'] = 40
    cfg.PARAMS['border'] = 10

    entity = gpd.read_file(get_demo_file('Hintereisferner.shp'))
    hef = workflow.init_glacier_regions(entity, reset=False)[0]

    input_shp = hef.get_filepath('outlines', div_id=0)
    input_dem = hef.get_filepath('dem', div_id=0)

    # set paths to python 2.7 and to the partitioning package
    python = 'path to python 2.7'
    project = 'path to the partitioning package'

    script = os.path.join(project, 'partitioning/run_divides.py')

    # run code from your console (PYTHON 2.7!)
    os.system(python + ' ' + script + ' ' + input_shp + ' ' + input_dem)

    print('Hintereisferner is divided into', hef.n_divides, 'parts.')

    tasks.glacier_masks(hef)
    tasks.compute_centerlines(hef)
    graphics.plot_centerlines(hef)
    plt.show()

