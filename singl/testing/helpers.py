import os
import pathlib
import numpy
from skimage import io


def create_test_csv_from_ids(image_ids, csv_file):
    with open(csv_file, 'w') as f:
        f.write(f'ID,ImageHeight,ImageWidth,Label')
        if isinstance(image_ids, str):
            image_ids = [image_ids]
        for iid in image_ids:
            f.write(f'\n{iid},10,10,0')


def create_test_images(image_ids, root_dir):
    pathlib.Path(root_dir).mkdir()
    if isinstance(image_ids, str):
        image_ids = [image_ids]
    for iid in image_ids:
        stems = ("_red.png", "_blue.png", "_yellow.png", "_green.png")
        channel = numpy.random.randint(0, 253, (10, 10), dtype=numpy.uint8)
        root = os.path.join(root_dir, iid)
        for stem in stems:
            io.imsave(root + stem, channel)
