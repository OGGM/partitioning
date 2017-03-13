import time
import shutil
import os
from partitioning import dividing_glaciers

if __name__ == '__main__':
    start0 = time.time()
    base_dir = '/home/juliaeis/Dokumente/OGGM/work_dir/CentralEurope/3000+'
    # base_dir = 'C:\\Users\\Julia\\OGGM_wd\\Alaska_non_tidewater'
    for dir in os.listdir(os.path.join(base_dir, 'per_glacier')):
        if dir.startswith('RGI50-11.00897') and not dir.endswith('.png'):
            print dir
        # if dir in ['RGI50-11.01144', 'RGI50-11.02460', 'RGI50-11.02755']:
            input_shp = base_dir+'/per_glacier/'+dir+'/outlines.shp'
            input_dem = os.path.dirname(input_shp)+'/dem.tif'
            for fol in os.listdir(os.path.dirname(input_shp)):
                if fol.startswith('divide'):
                    shutil.rmtree(os.path.dirname(input_shp)+'/'+fol)
            os.makedirs(base_dir+'/per_glacier/'+dir+'/divide_01')
            to_copy = [input_shp, os.path.dirname(input_shp)+'/outlines.shx',
                       os.path.dirname(input_shp)+'/outlines.dbf']
            for file in to_copy:
                shutil.copy(file, os.path.dirname(input_shp)+'/divide_01')

            n = dividing_glaciers(input_dem, input_shp)
