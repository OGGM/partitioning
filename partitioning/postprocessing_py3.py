import os
import geopandas as gpd
import numpy as np
import pandas as pd
import copy
from oggm.core.gis import multi_to_poly

def _merge_sliver(gpd_obj, polygon):
    """merge sliver polygon to the glacier with the longest shared boundary.
    If polygon does not touch the glaciers, False will be returned.

    Parameters
    ----------
    gpd_obj : gpd.GeoDataFrame
        contains the geometry of each glacier
    polygon : shapely.geometry.Polygon instance
        sliver polygon, which should be merged

    Returns
    -------
    new gpd.GeoDataFrame,
    bool
    """

    intersection_array = gpd_obj.intersection(polygon.boundary).length

    if np.max(intersection_array) != 0:
        max_b = np.argmax(intersection_array)
        poly = gpd_obj.loc[max_b, 'geometry'].simplify(0.01).buffer(0)
        geom = poly.union(polygon.buffer(0)).buffer(0)
        if geom.type is not 'Polygon':
            geom = geom.buffer(-0.01).buffer(0.01)
        gpd_obj.set_value(max_b, 'geometry', geom)
        merged = True
    # sliver does not touch glacier at the moment. Try again in the end
    else:
        merged = False
    return [gpd_obj, merged]


def _filter_divides(gpd_obj, filter_area, filter_alt_range,
                    filter_perc_alt_range):
    """ filter divides

    Parameters
    ----------
    gpd_obj                 : gpd.GeoDataFrame
    filter_area             : bool
                              (True: keep a divide only if it's area is not
                               smaller than 2% of the largest divide)
    filter_alt_range        : bool
                              (True: keep a divide only if the absolute
                               altitude range of the divide is larger than 100m
    filter_perc_alt_range   : bool
                              (True: keep a divide only if the altitude range
                               of the divide is larger than 10% of the glaciers
                               total altitude range

    Returns
    -------
    filtered gpd.GeoDataFrame object
    """

    # initialise nokeep
    nokeep = pd.Series(np.zeros(len(gpd_obj), dtype=np.bool))
    if filter_area is True:
        nokeep = nokeep | (gpd_obj['Perc_Area'] < 0.02)
    if filter_alt_range is True:
        nokeep = nokeep | (gpd_obj['Alt_Range'] < 100)
    if filter_perc_alt_range is True:
        nokeep = nokeep | (gpd_obj['Perc_Alt_R'] < 0.1)

    gpd_obj['keep'] = ~nokeep

    if np.sum(gpd_obj['keep']) in [0, 1]:
        # Nothing to do! The divide should be ignored
        return gpd_obj, False

    while not gpd_obj['keep'].all():
        geom = gpd_obj.loc[~gpd_obj['keep']].iloc[0]
        gpd_obj = gpd_obj.drop(geom.name)
        if geom.geometry.type == 'Polygon':
            gpd_obj, bool = _merge_sliver(gpd_obj, geom.geometry)
        else:
            for geo in geom.geometry:
                gpd_obj, bool = _merge_sliver(gpd_obj, geo)
    return gpd_obj, True


def remove_multipolygon(gpd_obj):
    '''
    try to change mulitpolygons to polygons
    Parameters
    ----------
    gpd_obj: GeoDataFrame containing the some MultiPolygons

    Returns  GeoDataFrame
    -------

    '''
    original = copy.deepcopy(gpd_obj)
    repair = gpd_obj[gpd_obj.type != 'Polygon']
    try:
        for i, geo in zip(repair.index, repair.geometry):
            area = np.array([g.area for g in geo])
            gpd_obj.set_value(i, 'geometry', geo[np.argmax(area)])
            indices = list(range(len(geo)))
            del indices[np.argmax(area)]
            for i in indices:
                gpd_obj, keep = _merge_sliver(gpd_obj, geo[i])
        return gpd_obj
    except:
        return original


def check_for_islands(gpd_obj, outline):
    '''
    check if some divides did not intersects with the exterior of the outline
    and removes them

    Parameters
    ----------
    gpd_obj : GeoDataFrame
    outline : outline of the glaciers

    Returns : GeoDataFrame without islands
    -------

    '''

    original = copy.deepcopy(gpd_obj)
    try:
        intersects = gpd_obj.intersects(outline.exterior.loc[0])
        to_merge = gpd_obj[~intersects].geometry
        gpd_obj = gpd_obj[intersects].reset_index()
        for geom in to_merge:
            gpd_obj, bool = _merge_sliver(gpd_obj, geom)
        return gpd_obj
    except:
        return original


def postprocessing(rgidf, gdir, filter):
    outline = gpd.read_file(gdir.get_filepath('outlines'))
    index = rgidf[rgidf['RGIId'] == gdir.rgi_id].index
    rgidf.loc[index, 'OGGM_Area'] = [float(outline.Area) / 10 ** 6]

    if os.path.exists(os.path.join(gdir.dir, 'divides.shp')):
        glaciers = gpd.read_file(os.path.join(gdir.dir, 'divides.shp'))
    else:
        return rgidf

    glaciers = remove_multipolygon(glaciers)
    glaciers = check_for_islands(glaciers, outline)
    glaciers = glaciers.to_crs(rgidf.crs)

    filter_area = False
    filter_alt_range = False
    filter_perc_alt_range = False
    if filter == 'alt':
        filter_alt_range = True
        filter_perc_alt_range = True
    elif filter == 'all':
        filter_area = True
        filter_alt_range = True
        filter_perc_alt_range = True

    try:
        glaciers, keep = _filter_divides(glaciers, filter_area,
                                         filter_alt_range,
                                         filter_perc_alt_range)
    except:
        keep = True
        rgidf.loc[index, 'remarks'] = ['no filter was used (error in merging)']

    divide = copy.deepcopy(outline)

    for i in range(len(glaciers)-1):
        divide = divide.append(outline, ignore_index=True)

    divide['Area'] = ""
    divide['remarks'] = " "
    if 'level_0' in glaciers.columns:
        glaciers = glaciers.drop('level_0', axis=1)
    glaciers = glaciers.reset_index()
    for i in glaciers.index:
        divide.loc[i]['geometry'] = glaciers.loc[i]['geometry']
    geo_is_ok = []
    new_geo = []

    # check geometry
    for g, a in zip(divide.geometry, glaciers.Area):

        if a < 0.01:
            geo_is_ok.append(False)
            continue

        try:
            new_geo.append(multi_to_poly(g))
            geo_is_ok.append(True)
        except:
            geo_is_ok.append(False)

    failed = divide.iloc[np.invert(geo_is_ok)]
    for i in failed.index:
        failed.loc[i, 'remarks'] = 'check_geometry failed'

    divide = divide.iloc[geo_is_ok]
    divide = divide.append(failed, ignore_index=False)

    # change RGIId
    new_id = [divide.loc[i, 'RGIId'] + '_d' + str(i + 1).zfill(2) for i
              in range(len(glaciers))]
    divide['RGIId'] = new_id

    for i in divide[divide.type != 'Polygon'].index:
        divide.loc[i, 'remarks'] = 'type != Polygon'

    divide = divide[~divide.geometry.is_empty]
    # check area
    divide['Area'] = ""

    cor_factor = float(outline.Area) / glaciers.Area.sum()

    if len(divide) < 2:
        rgidf.loc[index, 'remarks'] = [' is too small or has no valid divide']
    elif not keep:
        rgidf.loc[index, 'remarks'] = [' all divides were filtered']
    else:
        area_km = cor_factor * glaciers.Area

        # change cenlon,cenlat
        cenlon = [g.centroid.xy[0][0] for g in divide.geometry]
        cenlat = [g.centroid.xy[1][0] for g in divide.geometry]

        # write data
        divide.loc[divide.index, 'CenLon'] = cenlon
        divide.loc[divide.index, 'CenLat'] = cenlat
        # divide.loc[divide.index, 'geometry'] = new_geo
        divide.loc[divide.index, 'Area'] = area_km
        divide.loc[divide.index, 'OGGM_Area'] = glaciers.Area
        rgidf = rgidf.append(divide, ignore_index=True)

    if cor_factor > 1.2 or cor_factor < 0.8:
        rgidf.loc[index, 'remarks'] = [
            'sum of areas did not correlate with RGI_area']

    return rgidf
