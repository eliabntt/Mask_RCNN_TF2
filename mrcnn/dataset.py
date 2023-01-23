import os
import cv2
import numpy as np

import utils
from config import Config

class GRADEDataset(utils.Dataset):
    """
    Generates the GRADE synthetic dataset. 
    """
    def load_shapes(self, image_dir, mask_dir):
        self.add_class("grade", 1, "human")
        
        # Load all image file names
        img_fns = os.listdir(image_dir)
        
        for i in range(len(img_fns)):
            img_path = os.path.join(image_dir,img_fns[i])
            mask_path = os.path.join(mask_dir,img_fns[i][:-4]+'.npy') # similar name but in mask folder
            self.add_image("grade", image_id=i, path=img_path, mask_path=mask_path)

            
    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        if os.path.exists(info['mask_path']):
            masks = np.load(info['mask_path'], allow_pickle=True)
            mask = masks.item()['mask']
            classes = masks.item()['class']
            class_ids = np.array([self.class_names.index(s) for s in classes])
        else:
            mask = np.zeros((720, 960, 1))
            class_ids = np.array([0])
            
        return mask.astype(bool), class_ids.astype(np.int32)
    

class GRADEConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coco_min_load_resnet_3_stages_nobg"

    # Train on 1 GPU and 2 images per GPU. We can put multiple images on each
    # GPU because the images are small.
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    # IMAGE_RESIZE_MODE = 'square'
    # IMAGE_MIN_DIM = 720
    # IMAGE_MAX_DIM = 960
    
    # Use smaller anchors because our image and objects are small
    # RPN_ANCHOR_SCALES = (8*2, 16*2, 32*2, 64*2, 128*2)  # anchor side in pixels

    
    # USE_MINI_MASK = False
