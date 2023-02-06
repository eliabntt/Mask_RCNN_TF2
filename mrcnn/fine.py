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

# import coco tools
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco
from coco import evaluate_coco
from coco import CocoDataset

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")


# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# load your model
model_path = COCO_MODEL_PATH
# model_path = "/ps/project/irotate/GRADE_nets/maskrcnn/grade_blur_scratch/mask_rcnn_grade_blur_scratch_0249.h5"

# Define training configuration
config = GRADEConfig()
#config.LEARNING_RATE = 0.00001 # this is necessary with coco otherwise overfitting
config.display()

# Training dataset
dataset_train = GRADEDataset()
dataset_train.load_shapes('/media/ebonetto/WindowsData/maskrcnn/train/images', '/media/ebonetto/WindowsData/maskrcnn/train/masks')
dataset_train.prepare()

# Validation dataset // either using grade loader or coco
dataset_val = GRADEDataset()
dataset_val.load_shapes('/media/ebonetto/WindowsData/maskrcnn/val/images', '/media/ebonetto/WindowsData/maskrcnn/val/masks')
dataset_val.prepare()

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

model.load_weights(model_path, by_name=True, exclude="heads")
model.epoch=0 # reset epoch number

# Training
#print("Training network heads")
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=40,
            layers='heads'#,augmentation=augmentation
            )

# Training - Stage 2
# Finetune layers from ResNet stage 4 and up
#print("Fine tune Resnet stage 4 and up")
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
                        epochs=120,
            layers='4+'#,augmentation=augmentation
            )

# Training - Stage 3
# Fine tune all layers
print("Fine tune all layers")
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=160,
            layers='all'#,augmentation=augmentation
            )
