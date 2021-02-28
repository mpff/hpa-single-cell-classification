import os
import sys
import time
import base64
import zlib
import logging

import numpy
import pandas
import torch

import h5py
import skimage
import cv2
import json

import ctypes
import multiprocessing

from hpacellseg.utils import label_cell
from pycocotools import _mask as coco_mask


class SingleCellDataset(torch.utils.data.Dataset):
    """HPA images with cell segmentation masks and weak labels."""

    def __init__(self, csv_file, hdf5_file, transform=None, target_transform=None, test=False, n_samples=False):
        """
        Args:
            csv_file (string): Path to the csv file with cells with image_ids and labels.
            hdf_5file (string): DB with all the images.
            transform (callable, optional): Optional transform to be applied
                to the image in a sample.
            target_transform (callable, optional): 
            
        Returns:
            image (tensor), label (tensor), annotations (dict): A single cell sample.
        """
        self.frame = pandas.read_csv(csv_file, dtype = {
            'ID': str,
            'ImageHeigth' : int,
            'ImageWidth': int,
            'Label': str,
            'touches_edge_btlr': str,
            'RLEmask': bytes
        })
        if n_samples:
            self.frame = self.frame[:n_samples]
        self.hdf5_file = hdf5_file
        self.db = None
        self.transform = transform
        self.target_transform = target_transform
        self.test = test
        self.label_dict = {
            0: "Nucleoplasm", 1: "Nuclear membrane", 2: "Nucleoli",
            3: "Nucleoli fibrillar center", 4: "Nuclear speckles",
            5: "Nuclear bodies", 6: "Endoplasmic reticulum", 7: "Golgi apparatus", 
            8: "Intermediate filaments", 9: "Actin filaments", 10: "Microtubules", 
            11: "Mitotic spindle", 12: "Centrosome", 13: "Plasma membrane", 
            14: "Mitochondria", 15: "Aggresome", 16: "Cytosol", 
            17: "Vesicles and punctate cytosolic patterns", 18: "Negative"
        }
        
        # For Caching files in RAM:
        self._use_cache = False
        #self._cache = [None] * len(self.frame)
        shared_array_base = multiprocessing.Array(ctypes.c_float, len(self.frame)*4*224*224)
        shared_array = numpy.ctypeslib.as_array(shared_array_base.get_obj())
        shared_array = shared_array.reshape(len(self.frame), 4, 224, 224)
        self._image_cache = torch.from_numpy(shared_array)
        
        
    def __len__(self):
        return len(self.frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.db is None:
            self.db = h5py.File(self.hdf5_file, 'r')
        # Image identifier
        image_id = self.frame.iloc[idx, 0]
        image_height = self.frame.iloc[idx, 1]
        image_width = self.frame.iloc[idx, 2]
        touching_edges = self.frame.iloc[idx, 4]
        rle = self.frame.iloc[idx, 5]
        target, labels = self._create_label(idx)
        # Create annotations.
        annotations = {
            'ID': image_id,
            'ImageWidth': image_width,
            'ImageHeight': image_height,
            'rle': rle,
            'labels': labels,
            'touches': touching_edges
        }
        if self.target_transform:
            target = self.target_transform(target)
        # Load sample image from either cache or db.
        if not self._use_cache:
            image = self.db[image_id][...]  # Read from disk.
            image[...,[1, 2]] = image[...,[2, 1]]  # swap blue and yellow...
            sample = (image, target, annotations)
            if self.transform:
                sample = self.transform(sample)
            self._image_cache[idx] = sample[0]
        else:
            image = self._image_cache[idx]
            sample = (image, target, annotations)
        return sample[0], sample[1], sample[2]
    
    def set_use_cache(self):
        self._use_cache = True
    
    def _create_label(self, idx):
        label = self.frame.iloc[idx, 3]
        if self.test: label = "0|1"  # dummy label for test set.
        label = label.split('|')
        label = [int(i) for i in label]
        label_ohc = torch.zeros(len(self.label_dict))
        for i in range(len(self.label_dict)):
            if i in label:
                label_ohc[i] = 1.
        label_names = [self.label_dict[l] for l in label]
        return label_ohc, label_names

    
def collate_fn(batch):
    images, labels, annotations = zip(*batch)
    images = torch.stack(images)
    labels = torch.stack(labels)
    return images, labels, annotations


class HPASingleCellDataset(torch.utils.data.Dataset):
    """HPA single cell images with weak labels dataset."""

    def __init__(self, csv_file, root_dir, segmentator, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with image ids and labels.
            root_dir (string): Directory with all the images.
            segmentator (callable): Cell segmentator to mask single cells.
            transform (callable, optional): Optional transform to be applied
                to the image in a sample.
        """
        self.images_frame = pandas.read_csv(csv_file)
        self.root_dir = root_dir
        self.segmentator = segmentator
        self.transform = transform
        self.labels = ["Nucleoplasm", "Nuclear membrane", "Nucleoli",
                       "Nucleoli fibrillar center", "Nuclear speckles",
                       "Nuclear bodies", "Endoplasmic reticulum",
                       "Golgi apparatus", "Intermediate filaments",
                       "Actin filaments", "Microtubules", "Mitotic spindle",
                       "Centrosome", "Plasma membrane", "Mitochondria",
                       "Aggresome", "Cytosol", "Vesicles and punctate cytosolic patterns",
                       "Negative"]
        
    def __len__(self):
        return len(self.images_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Image identifier and (weak) class labels.
        image_id = self.images_frame.iloc[idx, 0]
        label = self.images_frame.iloc[idx, 1].split('|')
        label = [int(i) for i in label]
        # One hot encoding for labels.
        label_ohc = torch.zeros(19)
        label_ohc[label] = 1.
        # Load 4 channel image from path.
        base_path = os.path.join(self.root_dir, image_id)
        mt = io.imread(base_path + "_red.png")    
        er = io.imread(base_path + "_yellow.png")    
        nu = io.imread(base_path + "_blue.png")
        pt = io.imread(base_path + "_green.png")
        image = (mt, er, nu, pt)
        image = numpy.dstack(image)
        # Save image dimensions.
        id_shape = image.shape[:2]
        # Run HPACellSegmentator on image, get single cell masks.
        nuc_segmentations = self.segmentator.pred_nuclei([image[:,:,2]])  # Input has to be a list.
        cell_segmentations = self.segmentator.pred_cells([image[:,:,:3]], precombined=True)
        nuclei_mask, cell_mask = label_cell(nuc_segmentations[0], cell_segmentations[0])
        cell_ids = numpy.unique(cell_mask)
        cell_ids = cell_ids[1:]  # Drop background.
        masks = cell_mask == cell_ids[:, None, None]
        masks = [cell_mask == i for i in cell_ids]
        # Split image into single cells.
        images = []
        labels = []
        targets = []
        for mask in masks:
            target = {}
            target["image_id"] = image_id
            target["id_shape"] = id_shape
            target["rle"] = self.encode_binary_mask_(mask)
            target["label"] = label
            targets.append(target)
            labels.append(label_ohc)
            images.append(self.crop_image_with_mask_and_pad_to_square_(image, mask))
            if self.transform is not None:
                images[-1] = self.transform(images[-1])
        return [images, labels, targets]

    def encode_binary_mask_(self, mask):
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
    
    def crop_image_with_mask_and_pad_to_square_(self, image, mask):
        # Calculate bounding box.
        pos = numpy.where(mask)
        top = numpy.max(pos[0])
        left = numpy.min(pos[1])
        width = (numpy.max(pos[1]) - left) // 2 * 2  # easier padding to a square.
        height = (top - numpy.min(pos[0])) // 2 * 2
        # Crop image.
        image = image[top-height:top,left:left+width]
        mask = mask[top-height:top,left:left+width]
        image = image * mask[:,:,None]
        # Pad image to square symmetrically. What the hack?
        h, w, d = image.shape
        if h > w:
            p = int(0.5*(h-w))
            padding = ((0, 0), (p, p))
        else:
            p = int(0.5*(w-h))
            padding = ((p, p), (0, 0))
        image = [numpy.pad(image[...,i], padding, mode='constant', constant_values=0) for i in range(d)]
        image = numpy.dstack(image)
        return image
    
      
def collate_imagewise(batch):
    """ 
    data: is a list of tuples with (images, labels, targets) 
    returns: tuple of stacked tensors of images and labels
        and a list for targets
    """
    images = []
    labels = []
    annotations = []
    for i in range(len(batch)):
        images += batch[i][0]
        labels += batch[i][1]
        annotations += batch[i][2]
    images = torch.stack(images)
    labels = torch.stack(labels)

    return images, labels, annotations



class SingleCellDatasetKaggle(torch.utils.data.Dataset):
    """HPA images with cell segmentation masks and weak labels."""

    def __init__(self, csv_file, root_dir, transform=None, target_transform=None, test=False, n_samples=False):
        """
        Args:
            csv_file (string): Path to the csv file with cells with image_ids and labels.
            hdf_5file (string): DB with all the images.
            transform (callable, optional): Optional transform to be applied
                to the image in a sample.
            target_transform (callable, optional): 
            
        Returns:
            image (tensor), label (tensor), annotations (dict): A single cell sample.
        """
        self.frame = pandas.read_csv(csv_file, dtype = {
            'ID': str,
            'ImageHeigth' : int,
            'ImageWidth': int,
            'Label': str,
            'touches_edge_btlr': str,
            'RLEmask': bytes
        })
        if n_samples:
            self.frame = self.frame[:n_samples]
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.test = test
        self.label_dict = {
            0: "Nucleoplasm", 1: "Nuclear membrane", 2: "Nucleoli",
            3: "Nucleoli fibrillar center", 4: "Nuclear speckles",
            5: "Nuclear bodies", 6: "Endoplasmic reticulum", 7: "Golgi apparatus", 
            8: "Intermediate filaments", 9: "Actin filaments", 10: "Microtubules", 
            11: "Mitotic spindle", 12: "Centrosome", 13: "Plasma membrane", 
            14: "Mitochondria", 15: "Aggresome", 16: "Cytosol", 
            17: "Vesicles and punctate cytosolic patterns", 18: "Negative"
        }
        
        
    def __len__(self):
        return len(self.frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Image identifier
        image_id = self.frame.iloc[idx, 0]
        image_height = self.frame.iloc[idx, 1]
        image_width = self.frame.iloc[idx, 2]
        touching_edges = self.frame.iloc[idx, 4]
        rle = self.frame.iloc[idx, 5]
        target, labels = self._create_label(idx)
        # Create annotations.
        annotations = {
            'ID': image_id,
            'ImageWidth': image_width,
            'ImageHeight': image_height,
            'rle': rle,
            'labels': labels,
            'touches': touching_edges
        }
        if self.target_transform:
            target = self.target_transform(target)
        # Load 4 channel image from path.
        base_path = os.path.join(self.root_dir, image_id)
        mt = io.imread(base_path + "_red.png")    
        er = io.imread(base_path + "_yellow.png")    
        nu = io.imread(base_path + "_blue.png")
        pt = io.imread(base_path + "_green.png")
        image = (mt, er, nu, pt)
        sample = (image, target, annotations)
        if self.transform:
            sample = self.transform(sample)
        return sample[0], sample[1], sample[2]

    def _create_label(self, idx):
        label = self.frame.iloc[idx, 3]
        if self.test: label = "0|1"  # dummy label for test set.
        label = label.split('|')
        label = [int(i) for i in label]
        label_ohc = torch.zeros(len(self.label_dict))
        for i in range(len(self.label_dict)):
            if i in label:
                label_ohc[i] = 1.
        label_names = [self.label_dict[l] for l in label]
        return label_ohc, label_names
