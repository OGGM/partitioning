import time
import shutil
import os
from partitioning import dividing_glaciers
import pickle

if __name__ == '__main__':
    start0 = time.time()
    # base_dir = 'C:\\Users\\Julia\\OGGM_wd\\CentralEurope\\3000+'
    base_dir = 'C:\\Users\\Julia\\OGGM_wd\\Alaska_non_tidewater'
    RGI = pickle.load(open(base_dir+'\\10-15.pkl', 'rb'))
    for dir in os.listdir(base_dir+'\\per_glacier'):
        # if dir.startswith('RGI50-01.10008') and not dir.endswith('.png'):
        if dir in RGI and dir not in ['RGI50-01.00006', 'RGI50-01.00016', 'RGI50-01.00025', 'RGI50-01.00032']:
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
