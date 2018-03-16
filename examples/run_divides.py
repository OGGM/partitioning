import sys
sys.path.append('/home/users/julia/partitioning')
from partitioning.core import dividing_glaciers

if __name__ == '__main__':
    dividing_glaciers(input_shp=sys.argv[1], input_dem=sys.argv[2])
