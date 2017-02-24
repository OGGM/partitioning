
import salem
from oggm import workflow,cfg,graphics,tasks
import geopandas as gpd
import matplotlib.pyplot as plt
import pickle
# from partitioning import dividing_glaciers

if __name__ == '__main__':
    cfg.initialize()
    # gdirs without partitioning
    #base_dir = 'C:\\Users\\Julia\\OGGM_wd\\CentralEurope\\3000+'
    base_dir =  'C:\\Users\\Julia\\OGGM_wd\\Alaska_non_tidewater'
    cfg.PATHS['topo_dir']= 'C:\\Users\\Julia\\OGGM_wd\\topo_dir'
    RGI_FILE = base_dir + '\\01_rgi50_Alaska.shp'
    cfg.PATHS['working_dir'] = base_dir
    cfg.PARAMS['divides_gdf'] = gpd.GeoDataFrame()
    rgidf = salem.read_shapefile(RGI_FILE, cached=True)
    gdirs = workflow.init_glacier_regions(rgidf)
    ID_s = pickle.load(open(base_dir+'\\10-15.pkl','rb'))
    #ID_s = ['RGI50-01.10411']
    for gdir in gdirs:
        if gdir.n_divides is not 1 and gdir.rgi_id in ID_s:
            input_shp = gdir.get_filepath('outlines', div_id=0)
            input_dem = gdir.get_filepath('dem', div_id=0)
            # plot figure before partitioning
            tasks.glacier_masks(gdir)
            tasks.compute_centerlines(gdir)
            fig = plt.figure(figsize=(20, 10))
            ax0 = fig.add_subplot(1, 2, 2)
            graphics.plot_centerlines(gdir, ax=ax0)

            cfg.PATHS['working_dir'] = base_dir+'\\no_partioning'
            cfg.PARAMS['divides_gdf'] = gpd.GeoDataFrame()
            index = [i in [gdir.rgi_id] for i in rgidf.RGIId]
            gdirs_d = workflow.init_glacier_regions(rgidf.iloc[index])

            for gdir_d in gdirs_d:
                if gdir_d.rgi_id is gdir.rgi_id:
                    # n,k=dividing_glaciers(input_dem, input_shp)
                    tasks.glacier_masks(gdir_d)
                    tasks.compute_centerlines(gdir_d)

                    ax1 = fig.add_subplot(1, 2, 1)
                    graphics.plot_centerlines(gdir_d, ax=ax1)
            plt.savefig(base_dir+'\\plots' + '\\' + str(gdir_d.rgi_id) + '.png')
            #plt.show()

