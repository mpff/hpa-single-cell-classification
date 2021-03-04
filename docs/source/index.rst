.. rst-class:: hide-header

Welcome to my repo!
===================

Here I document my go at the HPA single cell classification challenge on Kaggle.


Installing the environment
--------------------------

Everything was developed using the anaconda Python environment::

    $ git clone mpff/hpa-singl
    $ cd hpa-singl
    $ conda env create -f environment-<cpu/cuda>.yml
    $ conda activate <env-name>

You can change ``<env-name>`` by editing the first line of the ``environment.yml`` file.


Running the Makefile
--------------------

By running the makefile you can reproduce the final submission (this will take a while)::

    $ make submission


Documentation
-------------

.. toctree::
   :maxdepth: 2

   scripts


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
