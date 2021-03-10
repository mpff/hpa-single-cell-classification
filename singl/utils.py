import os
import numpy
import pandas
from skimage import io


def read_ids_from_csv(csv_file):
    """ Reads a column named 'ID' from csv_file. This function was
        created to make sure basic I/O works in unit testing.
    """
    csv = pandas.read_csv(csv_file)
    return csv.ID


def read_hpa_image(image_id, root_dir):
    """ Reads a four channel HPA cell image given by 'image_id' from
        'root_dir' and returns it as a (H x W x 4) numpy array.
    """
    root = os.path.join(root_dir, image_id)
    stems = ("_red.png", "_blue.png", "_yellow.png", "_green.png")
    paths = [root+stem for stem in stems]
    image = [io.imread(path) for path in paths]
    return numpy.dstack(image)
