
import os
import sys
import numpy as np
import math

from shapely.geometry import mapping, shape
import rasterio
from rasterio.mask import mask
from pygeoprocessing import routing
import geopandas as gpd
from scipy.signal import medfilt

from skimage import img_as_float
from skimage.feature import peak_local_max
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import cascaded_union
import matplotlib.pyplot as plt


def _raster_mask(input_dem, polygon, out_name):
    """ mask a raster file along the polygon and saves the new raster

    Parameters
    ----------
    input_dem   : str
        path to the raster file
    polygon     : shapely.geometry.Polygon instance
        outline polygon
    out_name    : str
        name of the output raster

    Returns
    -------
    path to the output raster
    """

    output_dem = os.path.join(os.path.dirname(input_dem), out_name+'.tif')

    geoms = [mapping(polygon)]

    with rasterio.open(input_dem) as src:
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


def _compactness(polygon):
    """ check if polygon satisfy glacier compactness (Allen,1998)

    Parameters
    ----------
    polygon: shapely.geometry.Polygon instance

    Returns
    -------
    bool
    """
    coord = np.array(polygon.exterior.coords)

    y_min = Point(coord[np.argmin(coord[:, 1])])
    y_max = Point(coord[np.argmax(coord[:, 1])])
    x_min = Point(coord[np.argmin(coord[:, 0])])
    x_max = Point(coord[np.argmax(coord[:, 0])])
    # calculate max distance(perimeter)
    max_dist = y_min.distance(y_max)
    x_dist = x_min.distance(x_max)
    if x_dist > max_dist:
        max_dist = x_dist
    if max_dist*math.pi/polygon.boundary.length > 0.5:
        return True
    else:
        return False


def _fill_pits_with_saga(dem, saga_cmd=None):
    """ fill pits with SAGA GIS

    Parameters
    ----------
    dem     : str
        path to the raster file
    saga_cmd:   str
        path to saga_cmd.exe (only needen on windows system)

    Returns
    -------
    path to the new raster file
    """
    saga_dir = os.path.join(os.path.dirname(dem), 'saga')
    if not os.path.exists(saga_dir):
        # create folder for saga_output
        os.makedirs(saga_dir)
    saga_filled = os.path.join(saga_dir, 'filled.sdat')
    filled_dem = os.path.join(os.path.dirname(dem), 'filled_dem.tif')
    if sys.platform.startswith('linux'):
        os.system('saga_cmd ta_preprocessor 4 -ELEV:' + dem + ' -FILLED:'
                  + saga_filled + ' > /dev/null')
        os.system('gdal_translate ' + saga_filled + ' ' + filled_dem
                  + ' > /dev/null')
    elif sys.platform.startswith('win'):
        os.system('"'+saga_cmd+' ta_preprocessor 4 -ELEV:'+dem+' -FILLED:'
                  + saga_filled+' > /dev/null"')
        os.system('" gdal_translate '+saga_filled+' '+filled_dem
                  + ' > /dev/null"')
    return filled_dem


def _flowacc(input_dem):
    """create a raster which only contains flowaccumulation at the gutter,
    used for pour_point identification

    Parameters
    ----------
    input_dem   : str
        path to a raster file

    Returns
    -------
    path to raster file with the flow accumulation gutter
    """
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


def flowshed_calculation(dem, shp):
    """ calculate flowsheds

    Parameters
    ----------
    dem : str
        path to a raster file
    shp : str
        path to a shape file

    Returns
    -------
    path to the shape file containing all the flowsheds
    (shapely.geometry.Polygon)
    """
    dir = os.path.dirname(dem)
    watershed_dir = os.path.join(dir, 'all_watersheds.shp')
    flowshed_dir = os.path.join(dir, 'flowshed.shp')
    # calculate watersheds
    routing.delineate_watershed(dem, shp, 1, 100, watershed_dir,
                                os.path.join(dir, 'snapped_outlet_points.shp'),
                                os.path.join(dir, 'stream_out_uri.tif'))
    watersheds = gpd.read_file(watershed_dir).buffer(0)
    pour_points = gpd.read_file(dir + '/pour_points.shp')
    flowsheds = gpd.GeoSeries(watersheds.intersection(co), crs=crs)
    # remove empty objects

    flowsheds = flowsheds[(~flowsheds.is_empty) & (flowsheds.type != 'Point')
                          & (flowsheds.type != 'MultiPoint')
                          & (flowsheds.type != 'LineString')
                          & (flowsheds.type != 'MultiLineString')]
    collections = flowsheds[flowsheds.type == 'GeometryCollection']
    for index in collections.index:
        multi = MultiPolygon()
        for geo in collections.loc[index]:
            if geo.type is 'Polygon':
                multi = multi.union(geo)
        flowsheds.loc[index] = multi
    # if object is Multipolygon split it
    for i, shed in enumerate(flowsheds):
        if shed.type is 'MultiPolygon':
            # find polygon with minimal distance to pour point
            dist = []
            for s0 in shed:
                dist.append(s0.distance(pour_points.loc[i, 'geometry']))
            # add each polygon to all_watershed.shp
            for j, s1 in enumerate(shed):
                # polygon nearest to PP get current id
                if j == np.argmin(dist):
                    flowsheds.loc[i] = shape(s1)
                # all other poylgons were added at the end
                else:
                    s3 = gpd.GeoSeries(s1)
                    flowsheds = flowsheds.append(s3, ignore_index=True)

    result = gpd.GeoDataFrame(geometry=flowsheds)
    result.to_file(flowshed_dir)
    return flowshed_dir


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
    """ create flow accumulation gutter and identify pour points

    Parameters
    ----------
    dem :   str
        path to the raster file

    Returns
    -------
    path to a shape file containing all pour points
    """
    # calculation of flow accumulation and flow direction
    flow_gutter = _flowacc(dem)

    # identify Pour Points
    pour_points_shp = _pour_points(flow_gutter)

    return pour_points_shp


def _intersection_of_glaciers(gpd_obj, index):
    gpd_obj = gpd_obj.intersection(gpd_obj.loc[index, 'geometry'])
    gpd_obj = gpd.GeoDataFrame(geometry=gpd_obj[gpd_obj.index != index],
                               crs=crs)
    if not gpd_obj.empty:
        gpd_obj = _make_polygon(gpd_obj)
    return gpd_obj


def _create_p_glac(shp):
    pp = gpd.read_file(shp)
    geoms = [co.intersection(pp.loc[i, 'geometry'].buffer(
        _p_glac_radius(14.3, 0.5, pp.loc[i, 'flowacc']))) for i in pp.index]
    p_glac = gpd.GeoDataFrame(geometry=geoms, crs=crs)

    # delete empty objects
    p_glac = p_glac[~p_glac.is_empty]

    # if p_glac is Multipolygon choose only nearest polygon
    for i in p_glac.index:
        if p_glac.loc[i, 'geometry'].type is 'MultiPolygon':
            point = pp.loc[i, 'geometry']
            dist = [j.distance(point) for j in p_glac.loc[i, 'geometry']]
            min_dist = np.argmin(dist)
            p_glac.loc[i, 'geometry'] = p_glac.loc[i, 'geometry'][min_dist]
    return p_glac


def merge_flows(shed_shp, pour_point_shp):
    import time
    start = time.time()
    p_glac_dir = os.path.join(os.path.dirname(pour_point_shp), 'p_glac.shp')
    p_glac = _create_p_glac(pour_point_shp)
    p_glac.to_file(p_glac_dir)

    flows = gpd.read_file(shed_shp)
    # merge overlaps (p_glac, flowsheds)
    for j in p_glac.index:
        overlaps = flows[flows.intersects(p_glac.loc[j, 'geometry'])]

        if len(overlaps) > 1:
            union = cascaded_union(overlaps.loc[:, 'geometry'])
            #flows.loc[overlaps.index[0], 'geometry'] = union
            flows.set_value(overlaps.index[0], 'geometry', union)
            del_ids = overlaps.index.drop(overlaps.index[0])
            flows = flows.loc[flows.index.difference(del_ids)]

    # add gaps
    difference = co.difference(cascaded_union(flows.geometry))
    # gpd.GeoDataFrame(geometry=[difference], crs=crs).plot()
    if (difference.type is 'Polygon') and (difference.area > 0.1):
        glaciers, done = _merge_sliver(flows, difference)
    last_sliver = []
    if difference.type is 'MultiPolygon':
        for polygon in difference:
            if polygon.area > 0.1:
                glaciers, done = _merge_sliver(flows, polygon)
                if not done:
                    last_sliver.append(polygon)
    slivers = flows[flows.geometry.apply(_is_sliver)]
    glaciers = flows.loc[flows.index.difference(slivers.index)]
    if len(glaciers) <= 1:
        return 1
    glaciers.to_file(os.path.join(os.path.dirname(pour_point_shp), 'glaciers.shp'))
    slivers.to_file(os.path.join(os.path.dirname(pour_point_shp), 'slivers.shp'))
    # merge slivers to glaciers
    for k in slivers.index:
        sliver = slivers.loc[k, 'geometry']
        glaciers, done = _merge_sliver(glaciers, sliver)
        if not done:
            last_sliver.append(sliver)

    for polygon in last_sliver:
        glaciers, done = _merge_sliver(glaciers, polygon)

    # correct overlapping of glaciers
    for id in glaciers.index:
        glaciers = _merge_overlaps(glaciers, id)

    for id in glaciers.index:
        if id in glaciers.index:
            glaciers = _split_overlaps(glaciers, id)
    # save glaciers
    i = 1
    for id in glaciers.index:
        dir_name = 'divide_'+str(i).zfill(2)
        divide_shp = os.path.join(os.path.dirname(shed_shp), dir_name)
        if not os.path.isdir(divide_shp):
            os.makedirs(divide_shp)
        divide = out1.loc[0][:]
        divide.loc['geometry'] = glaciers.loc[id, 'geometry']
        divide.loc['Area'] = divide.geometry.area

        divide = gpd.GeoDataFrame([divide], geometry=[divide.geometry],
                                  crs=crs)
        divide = _make_polygon(divide)
        divide.to_file(os.path.join(divide_shp, 'outlines.shp'))
        i += 1
    print ('finish flows :', time.time()-start)
    return len(glaciers)

'''
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
        if shed.area < 100000 or (shed.area < 200000 and _compactness(shed)):
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
    print time.time()-start
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
        if glac.area < 100000 or (glac.area < 200000 and _compactness(glac)):
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
'''


def merge_sliver_poly(glacier_poly, polygon):
    max_b = 0
    max_b_id = -1
    for i, glac in glacier_poly.iteritems():
        if polygon.boundary.intersection(glac).length > max_b:
            max_b_id = i
            max_b = polygon.boundary.intersection(glac).length
    if not max_b_id == -1:
        glacier_poly[max_b_id] = glacier_poly[max_b_id].union(shape(polygon))
    return glacier_poly


def _make_polygon(gpd_obj):

    gpd_obj = gpd_obj[(gpd_obj.type == 'GeometryCollection') |
                      (gpd_obj.type == 'Polygon') |
                      (gpd_obj.type == 'MultiPolygon')]
    if not gpd_obj.empty:
        collection = gpd_obj[(~gpd_obj.is_empty) & (gpd_obj.type == 'GeometryCollection')]
        # choose only polygons or multipolygons
        for c in collection.index:
            geo = collection.loc[c, 'geometry']
            new = MultiPolygon()
            for obj in geo:
                if obj.type in ['Polygon', 'MultiPolygon']:
                    new = new.union(obj)
            gpd_obj = gpd_obj.copy()
            gpd_obj.loc[c, 'geometry'] = new
    return gpd_obj


def _p_glac_radius(a, b, f):
    if a * (f ** b) < 3500:
        return a * (f ** b)
    else:
        return 3500


def _is_sliver(polygon):

    if polygon.area < 100000 or (polygon.area < 200000 and _compactness(
            polygon)):
        return True
    else:
        return False


def _merge_sliver(gpd_obj, polygon):
    intersection_array = gpd_obj.intersection(polygon.boundary).length
    if np.max(intersection_array) != 0:
        max_b = np.argmax(intersection_array)
        geom = gpd_obj.loc[max_b, 'geometry'].union(polygon.buffer(0))
        gpd_obj.set_value(max_b, 'geometry', geom)
        merged = True
    # sliver does not touch glacier at the moment. Try again in the end
    else:
        merged = False
    return [gpd_obj, merged]


def _merge_overlaps(gpd_obj, l):

    if l in gpd_obj.index:
        inter = _intersection_of_glaciers(gpd_obj, l)
        # if intersection area  > 50%
        merge = inter[inter.area / gpd_obj.loc[l, 'geometry'].area > 0.5]
        while not merge.empty:
            if len(merge.index) == 1:
                gpd_obj.loc[l, 'geometry'] = gpd_obj.loc[l, 'geometry'].union(
                    gpd_obj.loc[merge.index[0], 'geometry'])
                gpd_obj = gpd_obj.loc[gpd_obj.index.difference(merge.index)]
                inter = _intersection_of_glaciers(gpd_obj, l)
                to_merge = inter.area / gpd_obj.loc[l, 'geometry'].area > 0.5
                merge = inter[to_merge]

            if len(merge.index) > 1:
                cascaded = cascaded_union(gpd_obj.loc[merge.index, 'geometry'])
                gpd_obj.loc[l, 'geometry'] = gpd_obj.loc[l, 'geometry']. \
                    union(cascaded)
                gpd_obj = gpd_obj.loc[gpd_obj.index.difference(merge.index)]
                inter = _intersection_of_glaciers(gpd_obj, l)
                to_merge = inter.area / gpd_obj.loc[l, 'geometry'].area > 0.5
                merge = inter[to_merge]
    return gpd_obj


def _split_overlaps(gpd_obj, l):

    split = _intersection_of_glaciers(gpd_obj, l)
    for k in split.index:
        if gpd_obj.loc[l, 'geometry'].area > gpd_obj.loc[k,
                                                         'geometry'].area:
            gpd_obj = _split_glacier(gpd_obj, k, split.loc[k, 'geometry'])
        else:
            gpd_obj = _split_glacier(gpd_obj, l, split.loc[k, 'geometry'])
    return gpd_obj


def _split_glacier(gpd_obj, index, polygon):
    diff = gpd_obj.loc[index, 'geometry'].difference(polygon)

    if diff.type is 'Polygon':
        if not _is_sliver(diff):
            gpd_obj.loc[index, 'geometry'] = diff
        else:
            gpd_obj = gpd_obj[gpd_obj.index != index]
            gpd_obj, done = _merge_sliver(gpd_obj, diff)

    else:
        # choose largest polygon
        max_poly = np.argmax([obj.area for obj in diff])
        if not _is_sliver(diff[max_poly]):
            gpd_obj.loc[index, 'geometry'] = diff[max_poly]
        else:
            gpd_obj, done = _merge_sliver(gpd_obj, diff[max_poly])
        # rest merged as sliver polygon
        rest = diff.difference(diff[max_poly])
        gpd_obj, done = _merge_sliver(gpd_obj, rest)
    return gpd_obj


def _smooth_dem(dem):
    smoothed_dem = os.path.join(os.path.dirname(dem), 'smoothed.tif')
    with rasterio.open(dem) as src:
        array = src.read()
        profile = src.profile
    # apply a 5x5 median filter to each band
    filtered = medfilt(array, (1, 5, 5)).astype('float32')
    with rasterio.open(smoothed_dem, 'w', **profile) as dst:
        dst.write(filtered)
    return smoothed_dem


def _transform_coord(tupel, transform):
    new_x = transform[0]+(tupel[1]+1)*transform[1]-transform[1]/2
    new_y = transform[3]+tupel[0]*transform[-1]-transform[1]/2

    return Point(new_x, new_y)


def _pour_points(dem):
    # open gutter with flow accumulation
    with rasterio.open(dem) as src:
        # TODO: new rasterio version will return affine.Affine()
        # --> order will change
        transform = src.transform
        band = np.array(src.read(1))
    im = img_as_float(band)
    nan = np.where(np.isnan(im))
    # set nan to zero
    im[nan] = 0
    # calculate maxima
    coordinates = peak_local_max(im, min_distance=1)
    # transform maxima to (flowaccumulation,coordinates)
    new_coord = []
    new = []
    dtype = [('flowaccumulation', float), ('coordinates', object)]
    # transform coordinates
    for x, y in coordinates:
        new_coord.append((im[x][y], _transform_coord([x, y], transform)))
        new.append(Point(_transform_coord([x, y], transform)))
    new_coord = np.array(new_coord, dtype=dtype)
    # sort array  by flowaccumulation
    new_coord = np.sort(new_coord, order='flowaccumulation')
    # reverse array
    new_coord = new_coord[::-1]
    pp = gpd.GeoDataFrame({'flowacc': new_coord['flowaccumulation']},
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

    # global outlines
    global crs
    global schema
    global pixelsize
    global co
    global out1
    pixelsize = 40

    smoothed_dem = _smooth_dem(dem)
    # fill pits
    filled_dem = _fill_pits_with_saga(smoothed_dem, saga_cmd=saga_cmd)

    # read outlines with gdal
    out1 = gpd.read_file(shp)
    crs = out1.crs
    co = out1.loc[0, 'geometry'].buffer(0)

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
    if sys.platform.startswith('win'):
        saga_cmd = 'C:\\"Program Files"\\SAGA-GIS\\saga_cmd.exe'
        gutter_dem = preprocessing(input_dem, input_shp, saga_cmd=saga_cmd)
    else:
        gutter_dem = preprocessing(input_dem, input_shp)
    pour_points_dir = identify_pour_points(gutter_dem)
    flowsheds_dir = flowshed_calculation(gutter_dem, pour_points_dir)
    no_glaciers = merge_flows(flowsheds_dir, pour_points_dir)

    # merge_flowsheds(p_glac, flowsheds_dir)

    # Allocation of flowsheds to individual glaciers
    # & Identification of sliver polygons
    # no_glaciers, all_polygon = merge_flowsheds(p_glac, watersheds)

    # delete files which are not needed anymore
    for file in os.listdir(os.path.dirname(input_shp)):
        for word in ['P_glac', 'all']:
            if file.startswith(word):
                os.remove(os.path.join(os.path.dirname(input_shp), file))
    return no_glaciers
