from partitioning import dividing_glaciers
import os
if __name__ == '__main__':
    base_dir='C:\Users\Julia\Dropbox\GlacierDir_Example'
    for dir in os.listdir(base_dir):
        if dir.startswith('RGI40-11.00897'):
            rgi_dir=base_dir+'/'+dir
            dividing_glaciers(rgi_dir+'/dem.tif',rgi_dir+'/outlines.shp')
