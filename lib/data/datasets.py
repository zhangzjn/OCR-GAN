"""
CREATE DATASETS
"""

# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

import torch.utils.data as data
import torch
from random import shuffle
from torchvision.datasets import DatasetFolder

from pathlib import Path
from PIL import Image
import numpy as np
import os
import os.path
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import cv2
from PIL import ImageFile

# pylint: disable=E1101

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class ImageFolder(data.Dataset):
    def __init__(self, root, transform=None):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        #self.noise = torch.FloatTensor(len(self.imgs), nz, 1, 1).normal_(0, 1)
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        # self.target_transform = target_transform
        # self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = cv2.imread(path)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        # latentz = self.noise[index]

        # TODO: Return these variables in a dict.
        # return img, latentz, index, target
        return img, target

    # def __setitem__(self, index, value):
    #     self.noise[index] = value

    def __len__(self):
        return len(self.imgs)

def FD(img):
    img = cv2.resize(img, (256,256))
    #img_resize = cv2.resize(img, (128,128))
    img_resize = cv2.pyrDown(img)
    temp_pyrUp = cv2.pyrUp(img_resize)
    #pdb.set_trace()
    temp_lap = cv2.subtract(img, temp_pyrUp)
    temp_lap = Image.fromarray(temp_lap)
    temp_pyrUp = Image.fromarray(temp_pyrUp)
    return temp_lap, temp_pyrUp

class ImageFolder_FD(data.Dataset):
    def __init__(self, root, transform=None):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        #self.noise = torch.FloatTensor(len(self.imgs), nz, 1, 1).normal_(0, 1)
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = cv2.imread(path)
        lap, res = FD(img)
        if self.transform is not None:
            lap = self.transform(lap)
            res = self.transform(res)

        # latentz = self.noise[index]

        # TODO: Return these variables in a dict.
        # return img, latentz, index, target
        return lap, res, target
    # def __setitem__(self, index, value):
    #     self.noise[index] = value

    def __len__(self):
        return len(self.imgs)

class ImageFolder_FD_Aug(data.Dataset):
    def __init__(self, root, transform=None, transform_aug=None):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        #self.noise = torch.FloatTensor(len(self.imgs), nz, 1, 1).normal_(0, 1)
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.transform_aug = transform_aug
        

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = cv2.imread(path)
        lap, res = FD(img)
        img = Image.fromarray(img)
        if self.transform is not None:
            fake_aug = self.transform_aug(img)
            lap = self.transform(lap)
            res = self.transform(res)

        # latentz = self.noise[index]

        # TODO: Return these variables in a dict.
        # return img, latentz, index, target
        return lap, res, fake_aug, target
    # def __setitem__(self, index, value):
    #     self.noise[index] = value

    def __len__(self):
        return len(self.imgs)
