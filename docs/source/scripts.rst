Scripts
*******

.. click:: singl.scripts.compress_dataset:compress_dataset
   :prog: compress_dataset

Dealing with the large number of images was a challenge, especially
as I had to store everything on quite slow network hdds.
I found that compressing all the images into a single file would
roughly decrease the read-in time for one image by ~60%. A huge
improvement!

I went with ``h5py`` for compression because the data could be
accesses dictionary style, which made it very natural to work with.
I decided to store every image as a single dataset, as it made images
very easy to access via their image ID, and did not seem to have any
drawbacks in terms of I/O speed (compared to a big combined numpy array
for all images of same resolution).

There are different compression types availalbe in ``h5py``. As the I/O
speed increase with ``lzf`` was huge compared to high or medium ``gzip``
compression and filesize was compareable, I went with ``lzf``.


Utility functions
-----------------

.. automodule:: singl.scripts.compress_dataset
    :members:
