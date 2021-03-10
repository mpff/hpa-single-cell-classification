import unittest
from click.testing import CliRunner
from singl.testing.helpers import create_test_csv_from_ids
from singl.testing.helpers import create_test_images

from singl.utils import read_ids_from_csv
from singl.utils import read_hpa_image


class TestIO(unittest.TestCase):

    def setUp(self):
        self.TEST_ID = "1a2b3c"
        self.TEST_IDS = ["1a", "2b", "3c"]
        self.TEST_DIR = "demo_dir"
        self.TEST_CSV = "demo.csv"

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


if __name__ == '__main__':
    unittest.main()
