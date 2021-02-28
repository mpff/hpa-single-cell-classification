import numpy
import torch
import skimage
import pycocotools
import json
import base64
import zlib
import ast


class Resize(object):
    """Resize 4 channel image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size, anti_aliasing=None):
        assert isinstance(output_size, (int, tuple))
        if anti_aliasing: assert isinstance(anti_aliasing, (bool))
        self.output_size = output_size
        self.anti_aliasing = anti_aliasing

    def __call__(self, sample):
        image, label, annotations = sample
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        
        rgb = skimage.transform.resize(image[...,:3], (new_h, new_w), anti_aliasing = self.anti_aliasing)
        y = skimage.transform.resize(image[...,3], (new_h, new_w), anti_aliasing = self.anti_aliasing)
        image = (rgb, y)
        image = numpy.dstack(image)
        return image, label, annotations

    
class MaskCrop(object):
    """Crop 4 channel image to bounding box of RLE encoded segmentation mask."""
    
    def __init__(self):
        pass
    
    def __call__(self, sample):
        image, target, annotations = sample
        rle = annotations['rle']
        height = annotations['ImageHeight']
        width = annotations['ImageWidth']
        encoded_mask = self.decode_b64_string(rle, height, width)
        # Decode RLE
        mask = pycocotools._mask.decode([encoded_mask])[:,:,0]
        bbox = pycocotools._mask.toBbox([encoded_mask])[0]
        x,y,w,h = (int(l) for l in bbox)
        mask = numpy.asfortranarray(mask)
        image = image[y:(y+h),x:(x+w),:]
        mask = mask[y:(y+h),x:(x+w)]
        image = image * mask[:,:,None]
        return image, target, annotations
    
    def decode_b64_string(self, rle, height, width):
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

    

class PadToSquare(object):
    """Pad 4channel image to square. Do not pad touching edges."""
    
    def __init__(self):
        pass
    
    def __call__(self, sample):
        image, target, annotations = sample
        h, w, d = image.shape
        touches = annotations['touches']
        touches = ast.literal_eval(touches)
        if h > w:
            p = h-w
            if touches[2] == True:
                padding = ((0, 0), (0, p))
            else:
                 padding = ((0, 0), (p, 0))
        else:
            p = w-h
            if touches[0] == True:
                padding = ((0, p), (0, 0))
            else:
                 padding = ((p, 0), (0, 0))
        image = [numpy.pad(image[...,i], padding, mode='constant', constant_values=0) for i in range(d)]
        image = numpy.dstack(image)
        return image, target, annotations

    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, target, annotations = sample

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        image = torch.tensor(image)
        return image, target, annotations
    
    
class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
   
   Args:
       mean (sequence): Sequence of means for each channel.
       std (sequence): Sequence of standard deviations for each channel.
       inplace(bool,optional): Bool to make this operation in-place.

    """

    def __init__(self,mean=[0.5,0.5,0.5,0.5],std=[0.5,0.5,0.5,0.5],imagewise=False):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
        self.imagewise = imagewise

    def __call__(self, sample):
        image, target, annotations = sample
        if self.imagewise:
            self.mean = torch.mean(image,0)
            self.std = torch.std(image, 0)
        image.sub_(self.mean.view(-1,1,1)).div_(self.std.view(-1,1,1))
        return image, target, annotations