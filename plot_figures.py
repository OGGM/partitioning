
import salem
from oggm import workflow,cfg,graphics,tasks
import matplotlib.pyplot as plt
import oggm
import os
import shutil
from partitioning import dividing_glaciers
import rasterio
if __name__ == '__main__':
    cfg.initialize()
    #gdirs without partitioning
    RGI=['RGI50-11.03418']
    base_dir='/home/juliaeis/Dokumente/OGGM/work_dir/CentralEurope/3000+'
    RGI_FILE = base_dir + '/outlines.shp'

    rgidf = salem.read_shapefile(RGI_FILE, cached=True)
    gdirs = workflow.init_glacier_regions(rgidf)
    for gdir in gdirs:
        if gdir.rgi_id in RGI:
            input_shp =gdir.get_filepath('outlines',div_id=0)
            input_dem=gdir.get_filepath('dem',div_id=0)

            for fol in os.listdir(gdir.dir):
                if fol.startswith('divide'):
                    shutil.rmtree(gdir.dir+'/'+fol)
            os.makedirs(gdir.dir+'/divide_01')
            for file in [input_shp,gdir.dir+'/outlines.shx',gdir.dir+'/outlines.dbf']:
                shutil.copy(file,gdir.dir+'/divide_01')

            #plot figure before partitioning
            tasks.glacier_masks(gdir)
            tasks.compute_centerlines(gdir)
            fig = plt.figure(figsize=(20,10))
            ax0 = fig.add_subplot(1, 2, 1)
            graphics.plot_centerlines(gdir, ax=ax0)

            n,k=dividing_glaciers(input_dem, input_shp)

            tasks.glacier_masks(gdir)
            tasks.compute_centerlines(gdir)

            ax1 = fig.add_subplot(1, 2, 2)

            graphics.plot_centerlines(gdir,ax=ax1)
            plt.savefig('/home/juliaeis/Dokumente/OGGM/work_dir/CentralEurope/plots_addition/'+str(gdir.rgi_id)+'.png')
            plt.show()

