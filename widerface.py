import os, sys
import cv2
import torch, torchvision
import detectron2
import scipy.io 

import numpy as np

from torch.utils.data import Dataset, DataLoader

class WIDERFACE(Dataset):
    def __init__(self, img_dir, phase, annot_file, transform=None):
        super(WIDERFACE, self).__init__()
        self.img_dir = img_dir
        self.phase = phase
        self.transform = transform
        self.data = self._load_annotations(annot_file)

    def _load_annotations(self, annot_file):
        """
        @param annot_file : absolute path of annotation text file
        @return : list of annotations (type: dictionary)

            annotation = {
                'file name' : image_full_path,
                'num_faces' : num_faces,
                'bboxes' : bboxes
            }
        
        """
        data = []
        with open (annot_file, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            image_path = lines[i].strip()
            image_full_path = os.path.join(self.img_dir, image_path)
            num_faces = int(lines[i + 1].strip())
            bboxes = [] # even 1 bbox when num_faces = 0 
            for j in range(num_faces):
                bbox = list(map(int, lines[i + 2 + j].strip().split()))
                bboxes.append(bbox)
            
            annotation = {
                'file name': image_full_path,
                'num_faces': num_faces,
                'bboxes': bboxes
            }
            
            data.append(annotation)
            i += max(2 + num_faces, 3) # 3 when num_faces = 0 
        
        return data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_full_path, num_faces, bboxes = self.data[index].values()
        image = cv2.imread(image_full_path)
        if image is None:
            raise ValueError(f"Image at path {image_full_path} not found!")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # BGR -> RGB
        bboxes = np.array(bboxes)
        
        if self.transform:
            image = self.transform(image)
        
        sample = {
            'image': image,
            'num_faces': num_faces,
            'bboxes': bboxes,
            } if self.phase != 'test' else {'image': image}

        return sample