from torch.utils.data import Dataset

import numpy as np
from PIL import Image

from xml.dom import minidom
from xml.etree.ElementTree import XML, fromstring

import os
import scipy.io as si
osp = os.path
osj = osp.join

class ImageNetLoader(Dataset):
    def __init__(self, path_images, csv_file, transform=None):
        self.path = path_images
        self.transform = transform
        # Read CSV file, skip empty lines and an optional header row
        with open(csv_file, 'r', encoding='utf-8') as data_obj:
            lines = [ln.strip() for ln in data_obj.readlines() if ln.strip()]

        # If the first line looks like a header (e.g., contains 'file' or 'file_name'), drop it
        if lines and ('file_name' in lines[0].lower() or 'file' in lines[0].lower() or 'filename' in lines[0].lower()):
            lines = lines[1:]

        self.listed_data = lines
        
    def __getitem__(self, idx):
        image_name, label = self.listed_data[idx].strip().split(',')
        image_ori = Image.open(osp.join(self.path,
                             image_name)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image_ori)

        return image,  int(label), image_name

    def __len__(self):
        return len(self.listed_data)  


