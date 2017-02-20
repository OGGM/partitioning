from oggm import workflow,cfg
import salem
if __name__ == '__main__':
    cfg.initialize()
    base_dir = 'C:\\Users\\Julia\\OGGM_wd\\CentralEurope\\1000-2000'
    #base_dir='/home/juliaeis/Dokumente/OGGM/work_dir/Alaska/land-terminating'
    cfg.PATHS['working_dir'] = base_dir
    #cfg.PATHS['dem_file']='/home/juliaeis/Dokumente/OGGM/work_dir/Alaska/dem.tif'
    RGI_FILE=base_dir+'/outlines.shp'
    RUN_DIVIDES=False

    rgidf = salem.read_shapefile(RGI_FILE, cached=True)
    gdirs = workflow.init_glacier_regions(rgidf)
    print (len(gdirs))
