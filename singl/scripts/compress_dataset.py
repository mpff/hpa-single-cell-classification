# -*- coding: utf-8 -*-
import click
import logging
import h5py
import singl


@click.command()
@click.argument('csv_file', type=click.Path(exists=True))
@click.argument('root_dir', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path(file_okay=True))
def compress_dataset(csv_file, root_dir, output_file):
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

    logger.info(f'Read image IDs from {csv_file}.')
    ids = singl.utils.read_ids_from_csv(csv_file)

    logger.info(f'Opening {output_file}.')
    with h5py.File(output_file, "w") as db:

        logger.info(f"Start compression. Iterating through image IDs.")
        for i, image_id in ids.iteritems():

            logger.info(f'Compressing {image_id} ({i}/{len(ids)})')
            image = singl.utils.read_hpa_image(image_id, root_dir)

            db.create_dataset(
                name=image_id,
                data=image,
                shape=image.shape,
                maxshape=image.shape,
                compression="lzf"  # very fast (however, compression is low!)
            )
