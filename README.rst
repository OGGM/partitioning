============
Partitioning
============

Many  glaciological applications(e.g `OGGM`_) use glacier outlines provided by the `Randolph Glacier
Inventory`_ (RGI).
In some cases these outlines represent a "glacier complex" and not those of a single glacier.
This results in incorrect calculations, especially as the model is developed to handle glaciers individually.

.. figure:: _pictures/RGI50-11.01791.png

Thus, a method seperating these complexes was developed by `Kienholz et al., (2013)`_. We have implemented this
method in the Python programming language and currently use `SAGA`_ and `GDAL`_ functions. All used software packages
are open source, making the algorithm developped by `Kienholz et al., (2013)`_ ,freely aviable.

The workflow as well as the suggested parameter values persist unmodifed.

In contrast to the original method we have to do another step at the end. As the pygeoprocessing package is still in development,
we obtained overlapping flowsheds from the watershed calculation. Hence, in a last step we check whether the received
glaciers overlap. If the overlapping area is greater than 50 percent of one of the glaciers, then
they are merged together. Otherwise, we allocate the intersection area to the larger glacier. This
eliminates all overlaps.

Requirements
------------
Software:

- Python 2.7 (caused by package dependencies)
- `GDAL`_
- `SAGA`_

Python packages:

- scipy
- numpy
- rasterio
- geopandas
- shapely
- skimage
- pygeoprocessing

Get the code
------------
The code of this project is available on GitHub. Clone the git repository::

    git clone https://github.com/OGGM/partitioning.git

Usage
-----

The required input data is a glacier outline of a single glacier and a digital elevation model (DEM) with a resolution of 40 m.
Note, that you can use `OGGM`_ to prepare the shapefile and the DEM for each glacier using a valid `RGI`_ file.

First example - Python 2.7
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
You can run the algorithm with the following lines:

.. code-block:: python

    import os
    from partitioning.core import dividing_glaciers

    # set paths to the required input files
    shp = os.path.join('path to dir', 'outlines.shp')
    dem = os.path.join('path to dir', 'dem.tif')

    #run dividing algorithm
    n = dividing_glaciers(input_shp=shp, input_dem=dem)
    print 'number of divides:', n

This creates automatically a shapefile containing all calculated divides.

Second example - Python 3 and OGGM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In a further example, we would like to show how to use the dividing algorithm together with `OGGM`_ in a Python 3 environment.
As we will run the dividing algorithm as an external program, you have to install a Python 2.7 version with the required packages (above).

We start with the usual first steps for OGGM:

.. code-block:: python

   import os

    from oggm import cfg,  workflow
    from oggm.utils import get_demo_file
    import matplotlib.pyplot as plt
    import geopandas as gpd

    if __name__ == '__main__':

        cfg.initialize()

        # Set Paths for OGGM
        cfg.PATHS['working_dir'] = 'path to working directory'

        # set dem resolution to 40 meters
        cfg.PARAMS['grid_dx_method'] = 'fixed'
        cfg.PARAMS['fixed_dx'] = 40
        cfg.PARAMS['border'] = 10
        cfg.PARAMS['use_intersects'] = False
        cfg.PARAMS['use_multiprocessing'] = False

        # get example shapefile initialize the model
        entity = gpd.read_file(get_demo_file('Hintereisferner_RGI6.shp'))
        hef = workflow.init_glacier_regions(entity, reset=False)[0]

        input_shp = hef.get_filepath('outlines')
        input_dem = hef.get_filepath('dem')

We can use the get_filepath function to get the required input data.

Next, we have to set the path to the Python 2.7 executable, where the pygeoprocessing package, as well as all the other required packages are installed. We also need the path from the partitioning package
to call the dividing algortihm from the console.


.. code-block:: python

    # set paths to python 2.7 and to the partitioning package
    python = 'path to python 2.7'
    project = 'path to the partitioning package'

    script = os.path.join(project, 'partitioning/examples/run_divides.py')

    # run code from your console (PYTHON 2.7!)
    os.system(python + ' ' + script + ' ' + input_shp + ' ' + input_dem)

    # reads the shapefile with the divides
    divides = gpd.read_file(os.path.join(hef.dir,'divides.shp'))

    print('Hintereisferner was divided in '+ str(len(divides)) + ' parts')


Postprocessing
~~~~~~~~~~~~~~

We developed additionally a postprocessing function, which contains a check for the geometry and the area, as well as some filter methods. This function creates a shapefile whcih contains
the same glaciers as the RGI files and addionally the shapes of the divides. This file can then be used by OGGM as a new input file and replaces the RGI-files.
Shapes that can't be corrected during the postprocessing, needs a manual correction and will have a remark at the output file.
We offer different filter methods:

- area filter               : keep a divide only if it's area is not smaller than 2% of the largest divide
- altutide filter           : keep a divide only if the absolute altitude range of the divide is larger than 100m
- percentual altitude filter: keep a divide only if the altitude range of the divide is larger than 10% of the glaciers total altitude range

Per default these all these filters are set to False. Calling the postprocessing function with the option filter='all', will set all methods to True.
The option filter='alt' only uses the altitude filter and the percentual altitude filter. An example for the Oetztal can be found `here`_


Get in touch
------------
Report bugs, share your ideas or view the source code on `GitHub`_.


.. _OGGM: http://oggm.readthedocs.io/en/latest/
.. _RGI: http://www.glims.org/RGI/
.. _Randolph Glacier Inventory: http://www.ingentaconnect.com/content/igsoc/jog/2014/00000060/00000221/art00012
.. _Kienholz et al., (2013): http://www.ingentaconnect.com/contentone/igsoc/jog/2013/00000059/00000217/art00011
.. _SAGA: http://www.saga-gis.org/en/index.html
.. _GDAL: http://www.gdal.org/
.. _GitHub: http://github.com/OGGM/partitioning
.. _here: https://github.com/OGGM/partitioning/blob/cluster/examples/oetztal.py