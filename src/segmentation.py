import os
import sys
import base64
import zlib
import glob

import numpy
import pandas
import torch
import matplotlib.pyplot as plt

from skimage import io
from hpacellseg.utils import label_cell
from pycocotools import _mask as coco_mask

import multiprocessing, logging
from joblib import Parallel, delayed
from tqdm import tqdm


def create_cell_segmentation_rles(csv_file, root_dir, segmentator,test=False,n_samples=None,n_workers=None):
    """Creates new maskwise test.csv."""
    if not n_workers: n_workers=num_cores
    images_frame = pandas.read_csv(csv_file)
    # Build image paths.
    print(f"Globbing {root_dir}.")
    fileiterator = glob.iglob(root_dir + '/' + '*_red.png')
    if n_samples:
        mt = [next(fileiterator) for i in range(n_samples)]
        print(f"Processing first {n_samples} images.")
    else:
        mt = [f for f in fileiterator]
        print(f"Processing {len(mt)} images.")
    er = [f.replace('red', 'yellow') for f in mt]
    nu = [f.replace('red', 'blue') for f in mt]
    images = zip(mt, er, nu)
    processed_list = Parallel(n_jobs=int(n_workers))(
        delayed(segment_image)(i, segmentator, images_frame, test) for i in tqdm(images)
    )
    print("Finished.")
    cells = []
    for l in processed_list:
        cells += l
    cells_frame = pandas.DataFrame(cells)
    return cells_frame

                                                
def segment_image(image, segmentator, images_frame, test):
    """ Returns list of cell masks. """
    mt, er, nu = image
    # Run segmentation model. (Takes list of lists...)
    nuc_segmentations = segmentator.pred_nuclei([nu])
    cell_segmentations = segmentator.pred_cells([[mt],[er],[nu]])
    # Get image ID and label. (hacky?)
    image_id = os.path.basename(mt).split('.')[0].replace("_red","")
    if test: 
        label = None
    else:
        print(image_id)
        print(images_frame.Label[images_frame.ID == image_id])
        label = images_frame.Label[images_frame.ID == image_id].values[0]
    # Get cell masks and rle encode.
    nuclei_mask, cell_mask = label_cell(nuc_segmentations[0], cell_segmentations[0])
    id_shape = cell_mask.shape
    cell_ids = numpy.unique(cell_mask)
    cell_ids = cell_ids[1:]  # Drop background.
    masks = [cell_mask == i for i in cell_ids]
    # Prepare output.
    cells = []
    for mask in masks:
        cell = {}
        cell['ID'] = image_id
        cell['ImageHeight'] = id_shape[0]
        cell['ImageWidth'] = id_shape[1]
        cell['Label'] = label
        cell['touches_edge_btlr'] = touches_edge(mask)
        cell['RLEmask'] = encode_binary_mask(mask)
        cells.append(cell)
    return cells
                                                

def encode_binary_mask(mask):
    """Converts a binary mask into OID challenge encoding ascii text."""
    # check input mask --
    if mask.dtype != numpy.bool:
        raise ValueError(
            "encode_binary_mask expects a binary mask, received dtype == %s" %
            mask.dtype)
    mask = numpy.squeeze(mask)
    if len(mask.shape) != 2:
        raise ValueError(
            "encode_binary_mask expects a 2d mask, received shape == %s" %
            mask.shape)
    # convert input mask to expected COCO API input --
    mask_to_encode = mask.reshape(mask.shape[0], mask.shape[1], 1)
    mask_to_encode = mask_to_encode.astype(numpy.uint8)
    mask_to_encode = numpy.asfortranarray(mask_to_encode)
    # RLE encode mask --
    encoded_mask = coco_mask.encode(mask_to_encode)[0]["counts"]
    # compress and base64 encoding --
    binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
    base64_str = base64.b64encode(binary_str)
    return base64_str


def decode_b64_string(rle, height, width):
    """Converts a rle mask into a binary mask."""
    # Data
    rle = rle[2:-1]
    # Decompress
    base64_str = rle.encode("utf8")
    binary_str = base64.b64decode(base64_str)
    encoded_mask = zlib.decompress(binary_str)
    # To RLE
    encoded_mask = {'counts': encoded_mask, 'size': (height, width)}
    return(encoded_mask)  