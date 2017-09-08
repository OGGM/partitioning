import sys
from partitioning.core import dividing_glaciers

if __name__ == '__main__':
    dividing_glaciers(input_shp=sys.argv[1], input_dem=sys.argv[2],
                      filter_area=sys.argv[3], filter_alt_range=sys.argv[4],
                      filter_perc_alt_range=sys.argv[5])