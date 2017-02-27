
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

from skimage import img_as_float
from skimage.feature import peak_local_max
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import cascaded_union


def _raster_clip(input_dem, shp, out_name, buffersize=0):

    output_dem = os.path.join(os.path.dirname(input_dem), out_name+'.tif')
    if os.path.basename(shp) == 'outlines.shp':
        geoms = [mapping(shape(outlines['geometry']).buffer(buffersize))]
    else:
        with fiona.open(shp, "r") as shapefile:
            geoms = [mapping(shape(shapefile.next()['geometry']).buffer(buffersize))]

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


def fill_pits_with_saga(saga_cmd, dem):
    saga_dir = os.path.join(os.path.dirname(dem), 'saga')

    if not os.path.exists(saga_dir):
        # create folder for saga_output
        os.makedirs(saga_dir)
    saga_filled = os.path.join(saga_dir, 'filled.sdat')
    filled_dem = os.path.join(os.path.dirname(dem), 'filled_dem.tif')
    if sys.platform.startswith('linux'):
        os.system('saga_cmd ta_preprocessor 4 -ELEV:' + dem + ' -FILLED:' + saga_filled)
        os.system('gdal_translate ' + saga_filled + ' ' + filled_dem)
    elif sys.platform.startswith('win'):
        os.system('"'+saga_cmd+' ta_preprocessor 4 -ELEV:'+dem+' -FILLED:'+saga_filled+'"')
        os.system('" gdal_translate '+saga_filled+' '+filled_dem+'"')
    return filled_dem


def flowacc(input_dem):
    new_flow_direction_map_uri = os.path.join(os.path.dirname(input_dem), 'flow_dir.tif')
    new_flow_map_accumulation_uri = os.path.join(os.path.dirname(input_dem), 'flow_accumulation.tif')
    # calculate flow direction
    routing.flow_direction_d_inf(input_dem, new_flow_direction_map_uri)
    # calculate flow_accumulation
    routing.flow_accumulation(new_flow_direction_map_uri, input_dem, new_flow_map_accumulation_uri)
    _raster_clip(new_flow_map_accumulation_uri, os.path.dirname(input_dem)+'/gutter.shp', 'flow_gutter')
    return os.path.dirname(input_dem)+'/flow_gutter.tif'


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


def gutter(masked_dem, depth):
    gutter_shp = os.path.join(os.path.dirname(masked_dem), 'gutter.shp')

    with fiona.open(gutter_shp, "w", "ESRI Shapefile", schema, crs) as output:
        outline_exterior = Polygon(shape(outlines['geometry']).exterior)
        # output.write({'properties': outlines['properties'],'geometry': mapping(shape(outlines['geometry']).buffer(pixelsize*2).difference(shape(outlines['geometry']).buffer(pixelsize)))})
        output.write({'properties': outlines['properties'], 'geometry': mapping(outline_exterior.buffer(pixelsize*2).difference(outline_exterior.buffer(pixelsize)))})
    gutter_dem = _raster_clip(masked_dem, gutter_shp, 'gutter')
    gutter2_dem = os.path.join(os.path.dirname(gutter_dem), 'gutter2.tif')
    with rasterio.open(masked_dem) as src1:
        mask_band = np.array(src1.read(1))
        with rasterio.open(gutter_dem) as src:
            mask_band = np.float32(mask_band - depth * (~np.isnan(np.array(src.read(1)))))
        with rasterio.open(gutter2_dem, "w", **src.meta.copy()) as dest:
            dest.write_band(1, mask_band)
    return gutter2_dem


def _my_small_func(para, option=False):
    """This simplifies the loop below.

    Parameters
    ----------
    para : float
        this is the number pi
    option : bool, optional
        this says yes or no
    Returns
    -------
    some variable
    """
    vara = para
    return vara


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


def dividing_glaciers(input_dem, input_shp):
    """ This is the main structure of the algorithm

    Parameters
    ----------
    input_dem : str
        path to the raster file(.tif) of the glacier, resolution have to be 40m!
    input_shp : str
        path to the shape file of the outline of the glacier

    Returns
    -------
    number of divides (int)
    """
    global outlines
    global crs
    global schema
    global pixelsize

    pixelsize = 40
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
    filled_dem = fill_pits_with_saga(saga_cmd, masked_dem)
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
