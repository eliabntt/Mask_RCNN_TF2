import os
import sys
import random
import math
import re
import time
import cv2
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)  # To find local version of the library

# Import Mask RCNN
import utils
import visualize
import model as modellib
from model import log
from dataset import GRADEDataset
from dataset import GRADEConfig

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco
from coco import evaluate_coco
from coco import CocoDataset

# Define training configuration
config = GRADEConfig()
config.LEARNING_RATE = 0.00001 # this is necessary otherwise overfitting
config.display()

# Training dataset
dataset_train = GRADEDataset()
dataset_train.load_shapes('/media/ebonetto/WindowsData/coco/coco2yolo/coco_red/train/images', '/media/ebonetto/WindowsData/coco/coco2yolo/coco_red/train/masks')
dataset_train.prepare()

#dataset_train = CocoDataset()
#coco = dataset_train.load_coco("/media/ebonetto/WindowsData/coco/coco2yolo/coco", "train", year=2017, return_coco=True, class_ids=[1], auto_download=False)
#dataset_train.prepare()

# Validation dataset
dataset_val = GRADEDataset()
dataset_val.load_shapes('/media/ebonetto/WindowsData/coco/coco2yolo/coco_red/val/images', '/media/ebonetto/WindowsData/coco/coco2yolo/coco_red/val/masks')
dataset_val.prepare()

dataset_val = CocoDataset()
coco = dataset_val.load_coco("/media/ebonetto/WindowsData/coco/coco2yolo/coco", "val", year=2017, return_coco=True, class_ids=[1], auto_download=False)
dataset_val.prepare()

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

# Train all
print("Fine tune all layers")
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=300,
            layers='all'#,augmentation=augmentation
            )
