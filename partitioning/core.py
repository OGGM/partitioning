
import os
import sys
import numpy as np
import math
import fiona
import shutil
from shapely.geometry import mapping, shape
import rasterio
from rasterio.tools.mask import mask
from pygeoprocessing import routing
import geopandas as gpd
import matplotlib.pyplot as plt

from skimage import img_as_float
from skimage.feature import peak_local_max
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import cascaded_union


def _raster_mask(input_dem, polygon, out_name):
    '''
    mask a raster file along the polygon, saves a new raster named out_name

    Parameters
    ----------
    input_dem   : str
        path to the raster file
    polygon     : Polygon
        polygon along the raster will
    out_name    : str
        name of the output raster (.tif)

    Returns
    -------
    path to the output raster
    '''

    output_dem = os.path.join(os.path.dirname(input_dem), out_name+'.tif')
    geoms = [mapping(polygon)]
    '''
    if os.path.basename(shp) == 'outlines.shp':
        geoms = [mapping(co.buffer(buffersize))]
        #geoms = [mapping(shape(outlines['geometry']).buffer(buffersize))]
    else:
        with fiona.open(shp, "r") as shapefile:
            geoms = [mapping(shape(shapefile.next()['geometry']).buffer(buffersize))]
    '''
    with rasterio.open(input_dem) as src:
        # out_image, out_transform = mask(src, geoms,nodata=np.nan, crop=False)
        out_image, out_transform = mask(src, geoms, nodata=np.nan, crop=False)
        out_meta = src.meta.copy()

    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "nodata": np.nan,
                     "transform": out_transform})
    with rasterio.open(output_dem, "w", **out_meta) as dest:
        dest.write(out_image)
    return output_dem


def compactness(polygon):
    coord = np.array(polygon.exterior.coords)
    # calculate max distance(perimeter)
    max_dist = Point(coord[np.argmin(coord[:, 1])]).distance(Point(coord[np.argmax(coord[:, 1])]))
    x_dist = Point(coord[np.argmin(coord[:, 0])]).distance(Point(coord[np.argmax(coord[:, 0])]))
    if x_dist > max_dist:
        max_dist = x_dist
    if max_dist*math.pi/polygon.boundary.length > 0.5:
        return True
    else:
        return False


def _fill_pits_with_saga(dem, saga_cmd=None):
    saga_dir = os.path.join(os.path.dirname(dem), 'saga')
    if not os.path.exists(saga_dir):
        # create folder for saga_output
        os.makedirs(saga_dir)
    saga_filled = os.path.join(saga_dir, 'filled.sdat')
    filled_dem = os.path.join(os.path.dirname(dem), 'filled_dem.tif')
    if sys.platform.startswith('linux'):
        os.system('saga_cmd ta_preprocessor 4 -ELEV:' + dem + ' -FILLED:'
                  + saga_filled)
        os.system('gdal_translate ' + saga_filled + ' ' + filled_dem)
    elif sys.platform.startswith('win'):
        os.system('"'+saga_cmd+' ta_preprocessor 4 -ELEV:'+dem+' -FILLED:'
                  + saga_filled+'"')
        os.system('" gdal_translate '+saga_filled+' '+filled_dem+'"')
    return filled_dem


def _flowacc(input_dem):
    flow_direction = os.path.join(os.path.dirname(input_dem), 'flow_dir.tif')
    flow_accumulation = os.path.join(os.path.dirname(input_dem),
                                     'flow_accumulation.tif')
    # calculate flow direction
    routing.flow_direction_d_inf(input_dem, flow_direction)
    # calculate flow_accumulation
    routing.flow_accumulation(flow_direction, input_dem, flow_accumulation)
    # mask along gutter
    gutter_shp = os.path.join(os.path.dirname(input_dem), 'gutter.shp')
    gutter = gpd.read_file(gutter_shp)['geometry'][0]
    _raster_mask(flow_accumulation, gutter, 'flow_gutter')

    return os.path.join(os.path.dirname(input_dem), 'flow_gutter.tif')


def flowsheds(input_dem):
    # open gutter with flow accumulation
    with rasterio.open(input_dem) as src:
        transform = src.transform
        band = np.array(src.read(1))
    im = img_as_float(band)
    nan = np.where(np.isnan(im))
    # set nan to zero
    im[nan] = 0
    # calculate maxima
    coordinates = peak_local_max(im, min_distance=4)
    # transform maxima to (flowaccumulation,coordinates)
    new_coord = []
    dtype = [('flowaccumulation', float), ('coordinates', np.float64, (2,))]
    # transform coordinates
    for coord in coordinates:
        new_coord.append((im[coord[0]][coord[1]], (transform[0]+(coord[1]+1)*transform[1]-transform[1]/2, transform[3] +
                                                   coord[0]*transform[-1]-transform[1]/2)))
    new_coord = np.array(new_coord, dtype=dtype)
    # sort array  by flowaccumulation
    new_coord = np.sort(new_coord, order='flowaccumulation')
    # reverse array
    new_coord = new_coord[::-1]

    with fiona.open(os.path.dirname(input_dem)+ '/P_glac.shp', "w", "ESRI Shapefile", {'geometry': 'MultiPolygon', 'properties': {'flow_acc': 'float', 'id': 'int'}}, crs) as p_glac:
        print 'start watershed calculation'
        # for each pour point: create shapefile and run delinate watershed
        with fiona.open(os.path.dirname(input_dem)+'/all_watershed.shp', "w", "ESRI Shapefile", {'geometry': 'MultiPolygon', 'properties': {'flow_acc': 'float', 'id': 'int'}}, crs) as all_watershed:
            i = 0
            m = len(new_coord)
            # print m
            while len(new_coord) is not 0:
                # create directory
                dir = os.path.dirname(input_dem)+'/'+str(i)
                if not os.path.isdir(dir):
                    os.makedirs(dir)

                coord = new_coord[0]
                # remove first element
                new_coord = new_coord[1:]
                # create radius around PPs and clip with outlines
                area = ({'properties': {'flow_acc': coord['flowaccumulation'], 'id': i}, 'geometry': mapping(shape(outlines['geometry']).intersection(shape(Point(coord['coordinates']).buffer(_p_glac_radius(14.3, 0.5, coord['flowaccumulation'])))))})
                # if result is Polygon, add it to shapefile
                if area['geometry']['type'] is 'Polygon':
                    p_glac.write(area)
                    # all_pourP.write({'properties': {'flow_acc': coord['flowaccumulation'], 'id': i,'p_glac': len(area['geometry']['coordinates'])},'geometry': {'type': 'Point', 'coordinates': coord['coordinates']}})
                # if result is MultiPolygon, add only the polygon whose perimeter is closest to PP
                elif area['geometry']['type'] is 'MultiPolygon':
                    min_dist = []
                    for j in shape(area['geometry']):
                        min_dist.append(shape(j).distance(Point(coord['coordinates'])))
                    area['geometry'] = mapping(shape(area['geometry'])[np.argmin(min_dist)])
                    p_glac.write(area)

                # write shapefile with ONE pour_point for watershed (transform unicode to ascii, otherwise segmentation fault)
                with fiona.open(dir+'/pour_point.shp', "w", "ESRI Shapefile", {'geometry': 'Point', 'properties': {'flow_acc':'float', 'id': 'int'}}, {k.encode('ascii'): v for k, v in crs.items()}) as output:
                    output.write({'properties': {'flow_acc': coord['flowaccumulation'], 'id': i}, 'geometry': {'type': 'Point', 'coordinates': coord['coordinates']}})

                # calculate watershed for pour point
                routing.delineate_watershed(os.path.dirname(input_dem)+'/gutter2.tif',dir+'/pour_point.shp', 0, 100, dir + '/watershed_out.shp', dir+'/snapped_outlet_points_uri.shp', dir+'/stream_out_uri.tif')

                # add watershed polygon to watershed_all.shp
                with fiona.open(dir+'/watershed_out.shp', "r", "ESRI Shapefile") as watershed:
                    w = watershed.next()
                # cut watershed with outlines
                w['geometry'] = mapping(shape(outlines['geometry']).intersection(shape(w['geometry']).buffer(0)))
                w['properties']['id'] = i

                if w['geometry']['type'] is 'Polygon':
                    all_watershed.write(w)
                if w['geometry']['type'] is 'MultiPolygon':
                    # find polygon with minimal distance to pour point
                    dist = []
                    for k in shape(w['geometry']):
                        dist.append(shape(k).distance(Point(coord['coordinates'])))
                    # add each polygon to all_watershed.shp
                    n = 0
                    for l in shape(w['geometry']):
                        w['geometry'] = mapping(shape(l))
                        # polygon nearest to PP get current id
                        if n == np.argmin(dist):
                            w['properties']['id'] = i
                        # all other poylgons get new id
                        else:
                            m += 1
                            w['properties']['id'] = m
                        all_watershed.write(w)
                        n += 1

                shutil.rmtree(dir)
                i += 1
        print 'watershed calculation finished'
    return [os.path.dirname(input_dem) + '/P_glac.shp', os.path.dirname(input_dem)+'/all_watershed.shp']


def _gutter(masked_dem, depth):
    """

    Parameters
    ----------
    masked_dem  : str
        path to a raster file
    depth       : int
        raster will be lowered by dept along gutter

    Returns
    -------
    path to the output raster

    """
    # create gutter shp
    gutter_shp = os.path.join(os.path.dirname(masked_dem), 'gutter.shp')
    outline_exterior = Polygon(co.exterior)
    gutter_shape = outline_exterior.buffer(pixelsize * 2).difference(
        outline_exterior.buffer(pixelsize))
    gpd.GeoSeries(gutter_shape, crs=crs).to_file(gutter_shp)

    # lower dem along gutter
    gutter_dem = _raster_mask(masked_dem, gutter_shape, 'gutter')
    gutter2_dem = os.path.join(os.path.dirname(gutter_dem), 'gutter2.tif')
    with rasterio.open(masked_dem) as src1:
        mask_band = np.array(src1.read(1))
        with rasterio.open(gutter_dem) as src:
            mask_band = np.float32(mask_band - depth * (~np.isnan(np.array(
                src.read(1)))))
        with rasterio.open(gutter2_dem, "w", **src.meta.copy()) as dest:
            dest.write_band(1, mask_band)

    return gutter2_dem


def identify_pour_points(dem):

    # calculation of flow accumulation and flow direction
    flow_gutter = _flowacc(dem)

    # identify Pour Points
    pour_points_shp = _pour_points(flow_gutter)
    return pour_points_shp


def merge_flowsheds(p_glac_dir, watershed_dir):
    import time
    start = time.time()

    sliver_poly = []
    watershed = []
    glacier_poly = {}
    total_glacier = MultiPolygon()
    p_poly = MultiPolygon()
    glacier_n = 0

    # determine overlaps from P_glac with watershed
    with fiona.open(watershed_dir, "r") as watersheds:
        for shed in watersheds:
            if not shape(shed['geometry']).is_valid:
                shed['geometry'] = shape(shed['geometry']).buffer(0)
            watershed.append(shape(shed['geometry']))

    with fiona.open(p_glac_dir, "r") as P_glac:
        for P in P_glac:
            p_poly = p_poly.union(shape(P['geometry']))
            to_merge = []
            for i, shed in enumerate(watershed):
                if shape(P['geometry']).intersects(shed):
                    to_merge.append(i)
                    watershed[to_merge[0]] = shape(watershed[to_merge[0]]).union(shed)
                    if watershed[to_merge[0]].type not in ['Polygon','MultiPolygon']:
                        new = MultiPolygon()
                        for g in watershed[to_merge[0]]:
                            if g.type in ['Polygon', 'MultiPolygon']:
                                new = new.union(g)
                        watershed[to_merge[0]] = new
            for shed in [watershed[x] for x in to_merge[1::]]:
                watershed.remove(shed)

    # check for sliverpolygons
    while len(watershed) is not 0:
        shed = watershed.pop()
        if shed.type != 'Polygon':
            for pol in shed[1::]:
                # if pol.type is 'Polygon':
                watershed.append(pol)
            shed = shed[0]
        if shed.area < 100000 or (shed.area < 200000 and compactness(shed)):
            sliver_poly.append(shed)
        else:
            glacier_poly.update({'glacier' + str(glacier_n): shed})
            glacier_n = glacier_n + 1
            total_glacier = total_glacier.union(shed)
    print len(glacier_poly), len(sliver_poly)
    if shape(outlines['geometry']).difference(total_glacier.buffer(0.01)).buffer(-0.2).type == 'Polygon':
        sliver_poly.append(shape(outlines['geometry']).difference(total_glacier.buffer(0.01)).buffer(-0.2).buffer(0.3))
    else:
        for gap in shape(outlines['geometry']).difference(total_glacier.buffer(0.01)).buffer(-0.2):
            sliver_poly.append(gap.buffer(0.3))

    for polygon in sliver_poly:
        glacier_poly = merge_sliver_poly(glacier_poly, polygon)

    with fiona.open(os.path.dirname(p_glac_dir) + '/glaciers.shp', "w", "ESRI Shapefile", schema, crs) as test:
        for g in glacier_poly:
            out = outlines['properties']
            out['Name'] = g
            test.write({'properties': out, 'geometry': mapping(glacier_poly[g])})

    from itertools import combinations
    inter = [[pair[0], pair[1]] for pair in combinations(glacier_poly.keys(), 2)]
    while len(inter) is not 0:

        key = inter.pop(0)
        if key[0] is not key[1]:
            intersection = (glacier_poly[key[0]].buffer(0)).intersection(glacier_poly[key[1]].buffer(0))
            if intersection.type in ['Polygon', 'MultiPolygon', 'GeometryCollection']:
                if intersection.type in ['GeometryCollection']:
                    poly = MultiPolygon()
                    for polygon in intersection:
                        if polygon.type in ['Polygon', 'Mulltipolygon']:
                            poly = poly.union(polygon)
                    intersection = poly
                if intersection.area / shape(glacier_poly[key[0]]).area > 0.5 or intersection.area / shape(
                        glacier_poly[key[1]]).area > 0.5:
                    # union of both glaciers
                    glacier_poly[key[0]] = shape(glacier_poly[key[0]]).union(glacier_poly[key[1]])
                    # delete 2nd glacier
                    for i, tupel in enumerate(inter):

                        if key[1] in tupel:
                            if tupel[0] is not tupel[1]:
                                inter[i].append(key[0])
                                inter[i].remove(key[1])

                    del glacier_poly[key[1]]
                elif shape(glacier_poly[key[0]]).area > shape(glacier_poly[key[1]]).area:
                    glacier_poly[key[1]] = (shape(glacier_poly[key[1]]).difference(glacier_poly[key[0]])).buffer(-0.1).buffer(0.1)
                    if glacier_poly[key[1]].type is 'MultiPolygon':
                        poly_max = Polygon()
                        for poly in glacier_poly[key[1]]:
                            if poly.area > poly_max.area:
                                if not poly_max.is_empty:
                                    glacier_poly[key[1]] = shape(glacier_poly[key[1]]).difference(poly_max)
                                    glacier_poly = merge_sliver_poly(glacier_poly, poly_max.buffer(0.1))
                                poly_max = poly
                            else:
                                glacier_poly[key[1]] = shape(glacier_poly[key[1]]).difference(poly)
                                glacier_poly = merge_sliver_poly(glacier_poly, poly.buffer(0.1))

                else:
                    glacier_poly[key[0]] = (shape(glacier_poly[key[0]].buffer(0)).difference(glacier_poly[key[1]])).buffer(-0.1).buffer(0.1)
                    # print glacier_poly[key[1]].type
                    if glacier_poly[key[0]].type is 'MultiPolygon':
                        poly_max = Polygon()
                        for poly in glacier_poly[key[0]]:
                            if poly.area > poly_max.area:
                                if not poly_max.is_empty:
                                    glacier_poly[key[0]] = shape(glacier_poly[key[0]]).difference(poly_max)
                                    glacier_poly = merge_sliver_poly(glacier_poly, poly_max.buffer(0.1))
                                poly_max = poly
                            else:
                                glacier_poly[key[0]] = shape(glacier_poly[key[0]]).difference(poly)
                                # print shape(glacier_poly[key[1]]).intersection(poly.buffer(0.1))
                                glacier_poly = merge_sliver_poly(glacier_poly, poly.buffer(0.1))
                                # print key[0] ,glacier_poly[key[0]].type, key[1], glacier_poly[key[1]].type

    # check if final_glaciers are not sliver polygon:
    keys = glacier_poly.keys()
    for glac_id in keys:
        glac = glacier_poly[glac_id]
        if glac.area < 100000 or (glac.area < 200000 and compactness(glac)):
            del glacier_poly[glac_id]
            glacier_poly = merge_sliver_poly(glacier_poly, glac)

    i = 1
    k = True
    for P in p_poly:
        no_merge = []
        for name in glacier_poly:
            if P.intersects(glacier_poly[name]):
                no_merge.append(name)
        if len(no_merge) > 1:
            glacier_poly[no_merge[0]] = cascaded_union([glacier_poly[x].buffer(0.1) for x in no_merge])
        for glacier in no_merge[1::]:
            glacier_poly.pop(glacier)

    for pol in glacier_poly:
        if not os.path.isdir(os.path.join(os.path.dirname(p_glac_dir), 'divide_'+str(i).zfill(2))):
            os.mkdir(os.path.join(os.path.dirname(p_glac_dir), 'divide_' + str(i).zfill(2)))
        with fiona.open(os.path.join(os.path.dirname(p_glac_dir), 'divide_' + str(i).zfill(2), 'outlines.shp'), "w",
                        "ESRI Shapefile", schema, crs) as gla:
            # for pol in glacier_poly
            if 'AREA' in schema['properties'].keys():
                outlines['properties']['AREA'] = glacier_poly[pol].area / 1000000
            elif 'Area' in schema['properties'].keys():
                outlines['properties']['Area'] = glacier_poly[pol].area / 1000000
            if glacier_poly[pol].type != 'Polygon':
                k = False
            gla.write({'properties': outlines['properties'], 'geometry': mapping(glacier_poly[pol])})
        i += 1
    return i - 1, k


def merge_sliver_poly(glacier_poly, polygon):
    max_boundary = 0
    max_boundary_id = -1
    for i, glac in glacier_poly.iteritems():
        if polygon.boundary.intersection(glac).length > max_boundary:
            max_boundary_id = i
            max_boundary = polygon.boundary.intersection(glac).length
    if not max_boundary_id == -1:
        glacier_poly[max_boundary_id] = glacier_poly[max_boundary_id].union(shape(polygon))
    return glacier_poly


def _p_glac_radius(a, b, f):
    if a * (f ** b) < 3500:
        return a * (f ** b)
    else:
        return 3500


def transform_coord(tupel, transform):
    new_x = transform[0]+(tupel[1]+1)*transform[1]-transform[1]/2
    new_y = transform[3]+tupel[0]*transform[-1]-transform[1]/2

    return Point(new_x, new_y)


def _pour_points(dem):
    # open gutter with flow accumulation
    with rasterio.open(dem) as src:
        transform = src.transform
        band = np.array(src.read(1))
    im = img_as_float(band)
    nan = np.where(np.isnan(im))
    # set nan to zero
    im[nan] = 0
    # calculate maxima
    coordinates = peak_local_max(im, min_distance=4)
    # transform maxima to (flowaccumulation,coordinates)
    new_coord = []
    new = []
    dtype = [('flowaccumulation', float), ('coordinates', object)]
    # transform coordinates
    for x, y in coordinates:
        new_coord.append((im[x][y], transform_coord([x, y], transform)))
        new.append(Point(transform_coord([x, y], transform)))
    new_coord = np.array(new_coord, dtype=dtype)
    # sort array  by flowaccumulation
    new_coord = np.sort(new_coord, order='flowaccumulation')
    # reverse array
    new_coord = new_coord[::-1]
    pp = gpd.GeoDataFrame({'flowaccumulation': new_coord['flowaccumulation']},
                          geometry=new_coord['coordinates'],  crs=crs)
    pp_shp = os.path.join(os.path.dirname(dem), 'pour_points.shp')
    pp.to_file(pp_shp)
    return pp_shp


def preprocessing(dem, shp, saga_cmd=None):

    """ Run all preprocessing tasks:

        fill pits from DEM,
        mask DEM along buffer1,
        lower it by 100 m along buffer2

    Parameters
    ----------
    dem     : str
        path to the DEM file
    shp     : str
        path to the shape file (outlines.shp)
    saga_cmd: str
        path to the SAGA GIS executable file (needed on win system)

    Returns
    -------
    path to the output raster
    """

    global outlines
    global crs
    global schema
    global pixelsize
    global co
    pixelsize = 40

    # fill pits
    filled_dem = _fill_pits_with_saga(dem, saga_cmd=saga_cmd)

    # read outlines.shp
    with fiona.open(shp, 'r') as shapefile:
        outlines = shapefile.next()
        # crs = shapefile.crs
        schema = shapefile.schema
        # outlines['geometry'] = shape(outlines['geometry']).buffer(0)

    # read outlines with gdal
    out = gpd.read_file(shp)
    crs = out.crs
    co = out['geometry'][0]
    # gpd.GeoSeries([co], crs=crs).to_file(os.path.join(os.path.dirname(shp),
    # 'co.shp'))

    # mask dem along buffer1
    masked_dem = _raster_mask(filled_dem, co.buffer(4*pixelsize), 'masked')

    # lower dem by l_gutter along gutter
    gutter_dem = _gutter(masked_dem, 100)

    return gutter_dem


def dividing_glaciers(input_dem, input_shp):
    """ This is the main structure of the algorithm

    Parameters
    ----------
    input_dem : str
        path to the raster file(.tif) of the glacier, resolution has to be 40m!
    input_shp : str
        path to the shape file of the outline of the glacier

    Returns
    -------
    number of divides (int)
    """

    gutter_dem = preprocessing(input_dem, input_shp)
    pour_points = identify_pour_points(gutter_dem)

    '''
    # read outlines.shp
    with fiona.open(input_shp, 'r') as shapefile:
        outlines = shapefile.next()
        crs = shapefile.crs
        schema = shapefile.schema
        if not shape(outlines['geometry']).is_valid:
            outlines['geometry'] = shape(outlines['geometry']).buffer(0)
    # read with gdal
    # outlines = gpd.read_file(input_shp)
    # outlines['geometry'] = outlines['geometry'].buffer(0)

    # clip dem along buffer1
    masked_dem = _raster_clip(input_dem, input_shp, 'masked', buffersize=4*pixelsize)

    # fill pits
    saga_cmd = 'C:\\"Program Files"\SAGA-GIS\saga_cmd.exe'
    filled_dem = fill_pits_with_saga(masked_dem, saga_cmd=saga_cmd)
    # TODO: when fill_pits from pygeoprocessing works, saga command could be replaced by the following 2 lines
    # filled_dem = os.path.dirname(masked_dem)+'//filled_dem.tif'
    # routing.fill_pits(masked_dem, filled_dem)

    # create gutter
    gutter_dem = gutter(filled_dem, 100)

    # calculation of flow accumulation and flow direction for PourPoint (PP)
    # determination
    flow_gutter = flowacc(gutter_dem)

    # identification of PP, watershed calculation
    [p_glac, watersheds] = flowsheds(flow_gutter)

    # Allocation of flowsheds to individual glaciers
    # & Identification of sliver polygons
    no_glaciers, all_polygon = merge_flowsheds(p_glac, watersheds)

    # delete files which are not needed anymore
    for file in os.listdir(os.path.dirname(input_shp)):
        for word in ['P_glac', 'flow', 'glaciers', 'all', 'gutter']:
            if file.startswith(word):
                os.remove(os.path.join(os.path.dirname(input_shp), file))
    return no_glaciers
    '''