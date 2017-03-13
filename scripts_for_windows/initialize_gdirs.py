from oggm import workflow,cfg
import salem
import geopandas as gpd
import pickle
if __name__ == '__main__':
    cfg.initialize()
    base_dir = '/home/juliaeis/Dokumente/OGGM/work_dir/CentralEurope/3000+'
    #base_dir = 'C:\\Users\\Julia\\OGGM_wd\\Alaska_non_tidewater'
    cfg.PATHS['working_dir'] = base_dir
    cfg.PATHS['topo_dir'] = '/home/juliaeis/Dokumente/OGGM/input_data/topo'
    cfg.PARAMS['divides_gdf'] = gpd.GeoDataFrame()
    # cfg.PATHS['dem_file'] = '/home/juliaeis/Dokumente/OGGM/work_dir/Alaska/dem.tif'
    RGI_FILE = base_dir+'/outlines.shp'
    # RGI_FILE = base_dir + '\\01_rgi50_Alaska.shp'

    # set dem to 40 meters
    cfg.PARAMS['d1'] = 40
    cfg.PARAMS['dmax'] = 40

    rgidf = salem.read_shapefile(RGI_FILE, cached=True)
    gdirs = workflow.init_glacier_regions(rgidf,reset=True)
    '''
    # only check tidewaterglaciers
    n = []
    for gdir in gdirs:
        if not gdir.is_tidewater and 10 < gdir.rgi_area_km2 < 15:
            print (gdir.rgi_id, gdir.terminus_type, gdir.rgi_area_km2)
            n.append(gdir.rgi_id)
    pickle.dump(n, open(str(base_dir + "\\10-15.pkl"), "wb"), protocol=2)
    print (len(n))
    '''