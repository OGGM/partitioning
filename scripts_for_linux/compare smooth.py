from oggm import workflow, cfg
import salem
import geopandas as gpd
import rasterio
import shutil
import os
import pickle
from partitioning.core import _fill_pits_with_saga, preprocessing, identify_pour_points
from scipy.signal import medfilt

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
    for gdir in gdirs:
        if gdir.rgi_id == 'RGI50-11.00002':
            dem = gdir.get_filepath('dem', div_id=0)
            outlines = gdir.get_filepath('outlines', div_id=0)

            with rasterio.open(dem) as src:
                array = src.read()
                profile = src.profile

            # apply a 5x5 median filter to each band
            filtered = medfilt(array, (1, 7, 7)).astype('float32')
            output = os.path.join(os.path.dirname(dem), 'median.tif')
            # Write to tif, using the same profile as the source
            with rasterio.open(output, 'w', **profile) as dst:
                dst.write(filtered)
            gutter = preprocessing(output, outlines)
            print gutter
            identify_pour_points(gutter)
