from oggm import workflow, cfg, tasks, graphics
from oggm.core.gis import _check_geometry
import salem
import geopandas as gpd
import pickle
import matplotlib.pyplot as plt
import subprocess
from shapely.geometry import Point
import os
import numpy as np
import copy


if __name__ == '__main__':

    cfg.initialize()

    filter_area = False
    filter_alt_range = False
    filter_perc_alt_range = False


    # check the paths
    base_dir = '/home/juliaeis/Dokumente/OGGM/work_dir/partitioning_v2'
    rgi_file = '/home/juliaeis/Dokumente/rgi60/05_rgi60_GreenlandPeriphery/05_rgi60_GreenlandPeriphery.shp'
    cfg.PATHS['topo_dir'] = '/home/juliaeis/Dokumente/OGGM/input_data/topo'
    cfg.PATHS['working_dir'] = base_dir
    cfg.PARAMS['use_intersects'] =False
    cfg.PARAMS['divides_gdf'] = gpd.GeoDataFrame()
    cfg.PARAMS['use_multiprocessing'] = False
    cfg.PARAMS['grid_dx_method'] = 'fixed'
    cfg.PARAMS['fixed_dx'] = 40
    cfg.PARAMS['border'] = 10
    cfg.PARAMS['continue_on_error']=True


    rgi = ['RGI60-05.14883']
    rgidf = salem.read_shapefile(rgi_file, cached=True)
    rgidf = rgidf[rgidf['RGIId'] == rgi[0]]

    #indices = [(i in rgi) for i in rgidf.RGIId]

    #gdirs = workflow.init_glacier_regions(rgidf, reset=False)
    gdirs = workflow.init_glacier_regions(rgidf, reset=False)

    all_divides = rgidf

    for gdir in gdirs:
        if gdir.rgi_id in rgi:
            input_shp = gdir.get_filepath('outlines')
            input_dem = gdir.get_filepath('dem')

            print(gdir.rgi_id)
            python = '/home/juliaeis/miniconda3/envs/test_pygeopro_env/bin/python'
            script = '/home/juliaeis/Documents/LiClipseWorkspace/partitioning-fork/test_altitude_filter/run_divides.py'
            #print(python+' ' + script + ' ' + input_shp + ' ' + input_dem + ' ' + filter_area + ' ' +filter_alt_range + ' ' + filter_perc_alt_range)

            #n = subprocess.call(python+' ' + script + ' ' + input_shp + ' ' + input_dem + ' ' +
             #                           str(filter_area) + ' ' + str(filter_alt_range) + ' ' + str(filter_perc_alt_range), shell=True) #+ ' > /dev/null')
            os.system(python + ' ' + script + ' ' + input_shp + ' ' + input_dem + ' ' +  str(filter_area) + ' ' + str(filter_alt_range) + ' ' + str( filter_perc_alt_range))  # + ' > /dev/null')
            #print(gdir.rgi_id+' is divided into '+str(int(n))+' parts')
            #except:
            #    print(gdir.rgi_id,'failed')

            outline = gpd.read_file(input_shp)

            index = rgidf[rgidf['RGIId'] == gdir.rgi_id].index
            rgidf.loc[index, 'OGGM_Area'] = [outline.area / 10 ** 6]

            if os.path.exists(os.path.join(os.path.dirname(input_shp), 'divides.shp')):
                glaciers = gpd.read_file(os.path.join(os.path.dirname(input_shp), 'divides.shp'))
            else:
                continue
            glaciers = glaciers.to_crs(rgidf.crs)

            divide = copy.deepcopy(outline)

            for i in range(len(glaciers) - 1):
                divide = divide.append(outline, ignore_index=True)

            divide['Area'] = ""
            divide['remarks'] = " "

            for i in range(len(glaciers)):
                divide.loc[i]['geometry'] = glaciers.loc[i]['geometry']

            # change RGIId
            new_id = [divide.loc[i, 'RGIId'] + '_d' + str(i + 1).zfill(2) for i
                      in range(len(glaciers))]
            divide.loc[divide.index, 'RGIId'] = new_id
            geo_is_ok = []
            new_geo = []


            # check geometry
            for g, a in zip(divide.geometry, glaciers.Area):

                if a < 0.01:
                    geo_is_ok.append(False)
                    continue

                try:
                    new_geo.append(_check_geometry(g))
                    geo_is_ok.append(True)
                except:
                    geo_is_ok.append(False)
                    rgidf.loc[index, 'remarks'] = ['geometry check failed']

            divide = divide.loc[geo_is_ok]
            # check area
            divide['Area'] = ""

            cor_factor = float(outline.Area) / glaciers.Area.sum()

            if len(divide) < 2:
                rgidf.loc[index, 'remarks'] = [gdir.rgi_id + ' is too small or has no valid divide...']

            elif cor_factor > 1.2 or cor_factor < 0.8:

                rgidf.loc[index, 'remarks'] = ['sum of areas did not correlate with RGI_area']

            else:
                area_km = cor_factor * glaciers.Area

                # change cenlon,cenlat
                cenlon = [g.centroid.xy[0][0] for g in divide.geometry]
                cenlat = [g.centroid.xy[1][0] for g in divide.geometry]

                # write data
                divide.loc[divide.index, 'CenLon'] = cenlon
                divide.loc[divide.index, 'CenLat'] = cenlat
                divide.loc[divide.index, 'geometry'] = new_geo
                divide.loc[divide.index, 'Area'] = area_km
                divide.loc[divide.index, 'OGGM_Area'] = glaciers.Area
                rgidf = rgidf.append(divide, ignore_index=True)
                print(divide.OGGM_Area.sum())
            print(outline.Area)
    sorted_rgi = rgidf.sort_values('RGIId')
    sorted_rgi = sorted_rgi[['Area', 'OGGM_Area', 'Aspect', 'BgnDate',
                             'CenLat', 'CenLon', 'Connect', 'EndDate', 'Form',
                             'GLIMSId', 'Linkages', 'Lmax', 'Name', 'O1Region',
                             'O2Region', 'RGIId', 'Slope', 'Status', 'Surging',
                             'TermType', 'Zmax', 'Zmed', 'Zmin', 'geometry',
                             'min_x', 'max_x', 'min_y', 'max_y', 'remarks']]

    sorted_rgi.to_file(os.path.join(base_dir, str(gdir.rgi_region)+'_DividedGlacierInventory.shp'))

