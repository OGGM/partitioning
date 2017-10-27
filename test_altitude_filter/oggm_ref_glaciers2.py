from oggm import workflow, cfg, tasks, graphics
from oggm.workflow import execute_entity_task
import salem
import geopandas as gpd
import pickle
import matplotlib.pyplot as plt
import subprocess
from shapely.geometry import Point
import os


if __name__ == '__main__':

    cfg.initialize()

    filter_area = False
    filter_alt_range = False
    filter_perc_alt_range = False


    # check the paths
    base_dir = '/home/juliaeis/Dokumente/OGGM/work_dir/partitioning_v2'
    rgi_file = '/home/juliaeis/Dokumente/rgi50/11_rgi50_CentralEurope/11_rgi50_CentralEurope.shp'
    cfg.PATHS['topo_dir'] = '/home/juliaeis/Dokumente/OGGM/input_data/topo'
    cfg.PATHS['working_dir'] = base_dir
    #
    cfg.PARAMS['divides_gdf'] = gpd.GeoDataFrame()
    cfg.PARAMS['use_multiprocessing'] = False
    cfg.PARAMS['grid_dx_method'] = 'fixed'
    cfg.PARAMS['fixed_dx'] = 40
    cfg.PARAMS['border'] = 10

    rgi = ['RGI50-11.00897']
    rgidf = salem.read_shapefile(rgi_file, cached=True)
    indices = [(i in rgi) for i in rgidf.RGIId]

    #gdirs = workflow.init_glacier_regions(rgidf, reset=False)
    gdirs = workflow.init_glacier_regions(rgidf[indices], reset=False)

    all_divides = rgidf

    for gdir in gdirs:
        if gdir.rgi_region in ['11']:
            input_shp = gdir.get_filepath('outlines', div_id=0)
            input_dem = gdir.get_filepath('dem', div_id=0)

            print(gdir.rgi_id)

            python = '/home/juliaeis/miniconda3/envs/test_pygeopro_env/bin/python'
            script = '/home/juliaeis/Documents/LiClipseWorkspace/partitioning-fork/test_altitude_filter/run_divides.py'
            #print(python+' ' + script + ' ' + input_shp + ' ' + input_dem + ' ' + filter_area + ' ' +filter_alt_range + ' ' + filter_perc_alt_range)

            n = subprocess.call(python+' ' + script + ' ' + input_shp + ' ' + input_dem + ' ' +
                                        str(filter_area) + ' ' + str(filter_alt_range) + ' ' + str(filter_perc_alt_range), shell=True) #+ ' > /dev/null')
            #print(gdir.rgi_id+' is divided into '+str(int(n))+' parts')
            #except:
            #    print(gdir.rgi_id,'failed')
            divides = salem.read_shapefile(os.path.join(gdir.dir, gdir.rgi_id+'_d.shp'))
            #divides = divides.to_crs(rgidf.crs)
            print(divides.crs)
            #points = gpd.GeoDataFrame(geometry=[Point(divides.loc[i, 'CenLat'], divides.loc[i, 'CenLon']) for i in divides.index])
            points.crs = divides.crs
            points = points.to_crs(rgidf)
            print(points)
            rgidf = rgidf.append(divides, ignore_index=True)
            print(rgidf.crs)
            sorted_rgi = rgidf.sort_values('RGIId')
   # print(rgidf['RGIId'])
    rgidf.to_file(os.path.join(base_dir, 'DividedGlacierInventory.shp'))