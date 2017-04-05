partitioning
------------

Many  glaciological applications(e.g OGGM) uses glacier outlines provided by the Randolph Glacier
Inventory (RGI) (`Pfeffer et al., (2014)`_]).

In some cases these outlines represent a "glacier complex" and not those of a single glacier.
This results in incorrect calculations especially as the model is developed to handle glaciers individually.

Thus, a method seperating these complexes was developed by `Kienholz et al., (2013)`_

.. image:: _pictures/RGI50-11.01791.png

.. _Pfeffer et al., (2014): http://www.ingentaconnect.com/content/igsoc/jog/2014/00000060/00000221/art00012
.. _Kienholz et al., (2013): http://www.ingentaconnect.com/contentone/igsoc/jog/2013/00000059/00000217/art00011