import unittest
import pathlib
import os
import numpy
import h5py
from skimage import io
from click.testing import CliRunner
from singl.scripts.compress_dataset import read_ids_from_csv
from singl.scripts.compress_dataset import read_hpa_image
from singl.scripts.compress_dataset import compress_dataset


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


class TestCompressDataset(unittest.TestCase):

    def setUp(self):
        self.TEST_ID = "1a2b3c"
        self.TEST_IDS = ["1a", "2b", "3c"]
        self.TEST_DIR = "demo_dir"
        self.TEST_CSV = "demo.csv"
        self.TEST_OUTPUT = "demo.hdf5"

    def test_read_ids_from_csv(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            create_test_csv_from_ids(self.TEST_ID, self.TEST_CSV)
            ids = read_ids_from_csv(self.TEST_CSV)
            self.assertEqual(ids[0], self.TEST_ID)

    def test_read_hpa_image(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            create_test_images(self.TEST_ID, self.TEST_DIR)
            image = read_hpa_image(self.TEST_ID, self.TEST_DIR)
            self.assertEqual(image.shape, (10, 10, 4))

    def test_compress_single_image(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            create_test_csv_from_ids(self.TEST_ID, self.TEST_CSV)
            create_test_images(self.TEST_ID, self.TEST_DIR)
            result = runner.invoke(compress_dataset,
                                   [self.TEST_CSV, self.TEST_DIR, self.TEST_OUTPUT])
            self.assertEqual(result.exit_code, 0)
            with h5py.File(self.TEST_OUTPUT, "r") as db:
                image = db[self.TEST_ID]
                self.assertEqual(image.shape, (10, 10, 4))

    def test_compress_dataset(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            create_test_csv_from_ids(self.TEST_IDS, self.TEST_CSV)
            create_test_images(self.TEST_IDS, self.TEST_DIR)
            result = runner.invoke(compress_dataset,
                                   [self.TEST_CSV, self.TEST_DIR, self.TEST_OUTPUT])
            self.assertEqual(result.exit_code, 0)
            with h5py.File(self.TEST_OUTPUT, "r") as db:
                image = db[self.TEST_IDS[0]]
                self.assertEqual(image.shape, (10, 10, 4))


if __name__ == '__main__':
    unittest.main()
