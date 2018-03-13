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
from oggm.workflow import execute_entity_task
from oggm.core.gis import _check_geometry,_polygon_to_pix
import pandas as pd
import matplotlib.patches as mpatches
import pickle



def make_panda(base_dir):
    all = pd.DataFrame(columns=['none', 'remarks', 2.0, 3.0, 4.0, 5.0, '6-10',
                                '11-15', '16-20', '21-30', '31-55'])

    for file in os.listdir(base_dir):

        if file.endswith('.shp'):
            print(file)
            rgi_file = os.path.join(base_dir, file)

            n = rgi_file.split('/')[-1].split('.')[0].split('_')
            name = n[0]+'-'+n[-1]

            cfg.PATHS['topo_dir'] = '/home/juliaeis/Dokumente/OGGM/input_data/topo'
            cfg.PATHS['working_dir'] = base_dir
            cfg.PARAMS['use_multiprocessing'] = True
            cfg.PARAMS['grid_dx_method'] = 'fixed'
            cfg.PARAMS['fixed_dx'] = 40
            cfg.PARAMS['border'] = 10
            cfg.PARAMS['continue_on_error'] = True
            dgi = gpd.read_file(rgi_file)
            '''
            failed = dgi[dgi.geometry.type != 'Polygon']
            #print(failed.RGIId)
            for g, i in zip(failed.geometry, failed.index):
                failed.set_value(i, 'geometry', _check_geometry(g))
                dgi.set_value(i, 'geometry', _check_geometry(g))


            gdirs = workflow.init_glacier_regions(dgi, reset=True)

            task_list = [
                tasks.glacier_masks,
                tasks.compute_centerlines]
            for task in task_list:
                execute_entity_task(task, gdirs)
            '''
            divides = dgi[dgi['RGIId'].str.find('_d') != -1]

            number = divides['RGIId'].str.split('_').apply(lambda x: x[0])
            count = number.value_counts().value_counts()

            list_count = {j: count[j] for j in
                          count[(count.index > 1) & (count.index < 6)].index}
            list_count['6-10'] = count[(count.index >= 6) & (count.index <= 10)].sum()
            list_count['11-15'] = count[(count.index >= 11) & (count.index <= 15)].sum()
            list_count['16-20'] = count[(count.index >= 16) & (count.index <= 20)].sum()
            list_count['21-30'] = count[(count.index >= 21) & (count.index <= 30)].sum()
            list_count['31-55'] = count[count.index > 30].sum()
            list_count['remarks']=[(len(dgi[dgi['remarks'] != '']))]

            all = all.append(
                pd.DataFrame(list_count, columns=list_count.keys(), index=[name]),
                ignore_index=False)
    return all.sort_index(ascending=False)


if __name__ == '__main__':

    cfg.initialize()
    #dir='/home/juliaeis/Schreibtisch/find_error/11_DividedGlacierInventory2.shp'
    #rgi = gpd.read_file(dir)
    '''
    base_dir_default = '/home/juliaeis/Schreibtisch/cluster/partitioning_results/alt_filter'

    base_dir_all = '/home/juliaeis/Schreibtisch/cluster/partitioning_results/all_filter'
    base_dir_no = '/home/juliaeis/Schreibtisch/cluster/partitioning_results/no_filter'

    alt_filter = make_panda(base_dir_default)
    all_filter = make_panda(base_dir_all)
    no_filter = make_panda(base_dir_no)

    no_filter['option'] = 'no filter'
    alt_filter['option'] = 'altitude filters only'
    all_filter['option'] = 'all filters'
    all_filter.loc[' '] = np.nan
    no_filter.loc[' '] = np.nan
    alt_filter.loc[' '] = np.nan
    # merged = pd.concat([no_filter,alt_filter])

    pickle.dump(no_filter, open('pd_no_filter.shp', 'wb'))
    pickle.dump(alt_filter, open('pd_alt_filter.shp', 'wb'))
    pickle.dump(all_filter, open('pd_all_filter.shp', 'wb'))

    no_filter = pickle.load(open('pd_no_filter.shp', 'rb'))
    alt_filter = pickle.load(open('pd_alt_filter.shp', 'rb'))
    all_filter = pickle.load(open('pd_all_filter.shp', 'rb'))

    index_dir = '/home/juliaeis/Schreibtisch/cluster/no_filter'
    new_index=[]
    for file in os.listdir(index_dir):
        if file.endswith('.shp'):
            n = file.split('/')[-1].split('.')[0].split('_')
            new_index.append(n[0]+'-'+n[-1])
    new_index.append(' ')
    new_index = np.flip(np.sort(new_index), axis=0)
    all_filter.index = new_index

    ax = no_filter.sort_index(ascending=False).plot.barh(width=0.25,
                                                         fontsize=8,
                                                         stacked=True,
                                                         figsize=(20, 10),
                                                         position=-0.5)
    handles, labels = ax.get_legend_handles_labels()
    alt_filter.sort_index(ascending=False).plot.barh(ax=ax, width=0.25,
                                                     fontsize=8, stacked=True,
                                                     figsize=(20, 10),
                                                     position=0.5, alpha=0.75,
                                                     legend=False)
    all_filter.sort_index(ascending=False).plot.barh(ax=ax, width=0.25,
                                                     fontsize=8, stacked=True,
                                                     figsize=(20, 10),
                                                     position=1.5, alpha=0.5,
                                                     legend=False)

    ax = plt.gca().add_artist(plt.legend(handles[1:], labels[1:]))
    normal_patch = mpatches.Patch(color='orange',
                                  label='partitioning without filter')
    trans_patch = mpatches.Patch(color='orange', alpha=0.75,
                                 label='altitude filters')
    trans_patch1 = mpatches.Patch(color='orange', alpha=0.5,
                                  label='all filters')
    plt.legend(handles=[normal_patch, trans_patch, trans_patch1])
    plt.title('number of divides per RGI region')
    plt.xlabel('absolute number of glaciers')
    plt.ylabel('RGI-region')

    #plt.show()
    '''
    #check valit

    index_dir = '/home/juliaeis/Schreibtisch/cluster/partitioning_results/no_filter'
    for file in os.listdir(index_dir):
        if file.endswith('.shp') and file.startswith('05'):
            rgi = os.path.join(index_dir, file)
    rgidf = salem.read_shapefile(rgi, cached=True)
    glacier = rgidf[rgidf['RGIId'].apply(lambda x: str(x).startswith('RGI60-05.10395_d'))]
    outline = salem.read_shapefile('/home/juliaeis/Dokumente/OGGM/work_dir/partitioning_v2/per_glacier/RGI60-05/RGI60-05.10/RGI60-05.10395/outlines.shp')
    #print(glacier.intesects(outline.exterior.coords))
    print(outline.Area)
    print(glacier)
    glacier.plot()
    plt.show()



    #gdirs = workflow.init_glacier_regions(rgidf, reset=True)