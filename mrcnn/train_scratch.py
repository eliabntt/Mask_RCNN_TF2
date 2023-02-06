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

# Define training configuration
config = GRADEConfig()
config.display()

# Training dataset
dataset_train = GRADEDataset()
dataset_train.load_shapes('/media/ebonetto/WindowsData/GRADE_DATASET/train/images', '/media/ebonetto/WindowsData/GRADE_DATASET/train/masks')
dataset_train.prepare()

# Validation dataset
dataset_val = GRADEDataset()
dataset_val.load_shapes('/media/ebonetto/WindowsData/GRADE_DATASET/valid/images', '/media/ebonetto/WindowsData/GRADE_DATASET/valid/masks')
dataset_val.prepare()

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

print("Train all layers")
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=300,
            layers='all' #,augmentation=augmentation
            )
