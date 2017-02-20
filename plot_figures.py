
import salem
from oggm import workflow,cfg,graphics,tasks
import matplotlib.pyplot as plt
import oggm
import os
import shutil
#from partitioning import dividing_glaciers
import rasterio
if __name__ == '__main__':
    cfg.initialize()
    #gdirs without partitioning
    RGI=['RGI50-11.00897']
    base_dir='C:\\Users\\Julia\\OGGM_wd\\CentralEurope\\3000+'
    RGI_FILE = base_dir + '/outlines.shp'
    cfg.PATHS['working_dir'] = base_dir+'\\no_partioning'
    rgidf = salem.read_shapefile(RGI_FILE, cached=True)
    gdirs = workflow.init_glacier_regions(rgidf)
    for gdir in gdirs:
        if gdir.rgi_id in RGI:
            input_shp =gdir.get_filepath('outlines',div_id=0)
            input_dem=gdir.get_filepath('dem',div_id=0)
            #plot figure before partitioning
            tasks.glacier_masks(gdir)
            tasks.compute_centerlines(gdir)
            fig = plt.figure(figsize=(20,10))
            ax0 = fig.add_subplot(1, 2, 1)
            graphics.plot_centerlines(gdir, ax=ax0)

            cfg.PATHS['working_dir']=base_dir
            index=[i in [gdir.rgi_id] for i in rgidf.RGIId]
            #print (rgidf.iloc[index])
            gdirs_d = workflow.init_glacier_regions(rgidf.iloc[index])

            for gdir_d in gdirs_d:
                if gdir_d.rgi_id is gdir.rgi_id:
                    #n,k=dividing_glaciers(input_dem, input_shp)

                    tasks.glacier_masks(gdir_d)
                    tasks.compute_centerlines(gdir_d)

                    ax1 = fig.add_subplot(1, 2, 2)

                    graphics.plot_centerlines(gdir_d,ax=ax1)
            #plt.savefig(gdir_d.dir+'\\'+str(gdir_d.rgi_id)+'.png')
            plt.show()

