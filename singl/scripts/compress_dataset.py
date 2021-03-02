# -*- coding: utf-8 -*-
import click
import logging
import os

import skimage
import numpy
import pandas

import h5py



def read_hpa_image(image_id, root_dir):
    """ Reads a four channel HPA cell image given by 'image_id' from
        'root_dir' and returns it as a (H x W x 4) numpy array.
    """
    root = os.path.join(root_dir, image_id)
    stems = ("_red.png", "_blue.png", "_yellow.png", "_green.png")
    paths = [root+stem for stem in stems]
    image = [skimage.io.imread(path) for path in paths]
    return numpy.dstack(image)


@click.command()
@click.argument('csv_file', type=click.Path(exists=True))
@click.argument('root_dir', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path(file_okay=True))
def main(csv_file, root_dir, output_file):
    """ Runs a data compression script to turn raw images specified by their
        file id in CSV_FILE and located in ROOT_DIR into a compressed .hdf5
        file saved as OUTPUT_FILE.
        This significantly speeds up I/O operations on potentially tens of
        thousands of images and helps keep the filesystem clean (having 80k
        individual files in a single directory is generally NOT advisable).

        Makes some assumptions about file naming convention and number of
        channels per image, as given by the HPA Single Cell Classification
        challenge on Kaggle.
    """
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    logger.info('Compressing images into single hd5f file.')

    logger.info(f'Reading {csv_file}')
    csv = pandas.read_csv(csv_file)

    logger.info(f'Grabbing "ID" column.')
    ids = csv.ID

    logger.info(f'Opening {output_file}.')
    with h5py.File(output_file, "w") as db:

        logger.info(f"Start compression. Iterating through image IDs.")
        for i, image_id in ids.iteritems():

            logger.info(f'Compressing {image_id} ({i}/{len(ids)})')
            image = read_hpa_image(image_id, root_dir)

            db.create_dataset(
                name=image_id,
                data=image,
                shape=image.shape,
                maxshape=image.shape,
                compression="lzf"  # very fast (however, compression is low!)
            )