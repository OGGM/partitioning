============
Partitioning
============

Many  glaciological applications(e.g `OGGM`_) uses glacier outlines provided by the `Randolph Glacier
Inventory`_ (RGI).
In some cases these outlines represent a "glacier complex" and not those of a single glacier.
This results in incorrect calculations, especially as the model is developed to handle glaciers individually.

.. figure:: _pictures/RGI50-11.01791.png

    centerlines of RGI50-11.01791 computed with outlines provided by RGI(left) and for each single glacier after dividing(right)

Thus, a method seperating these complexes was developed by `Kienholz et al., (2013)`_. We have implemented this
method in the Python programming language and currently use `SAGA`_ and `GDAL`_ functions. Compared to the
described algorithm in `Kienholz et al., (2013)`_ , all used software packages are open source.

The workflow as well as the suggested parameter values persist unmodifed.

In contrast to the original method we have to do another step at the end. As the pygeoprocessing package is still in development,
we obtained overlapping flowsheds from the watershed calculation. Hence, in a last step we check whether the received
glaciers overlap. If the overlapping area is greater than 50 percent of one of the glaciers, then
they are merged together. Otherwise, we allocate the intersection area to the larger glacier. This
eliminates all overlaps.

Requirements
------------
software:

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

.. _OGGM: http://oggm.readthedocs.io/en/latest/
.. _Randolph Glacier Inventory: http://www.ingentaconnect.com/content/igsoc/jog/2014/00000060/00000221/art00012
.. _Kienholz et al., (2013): http://www.ingentaconnect.com/contentone/igsoc/jog/2013/00000059/00000217/art00011
.. _SAGA: http://www.saga-gis.org/en/index.html
.. _GDAL: http://www.gdal.org/