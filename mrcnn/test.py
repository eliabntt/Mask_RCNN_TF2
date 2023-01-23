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
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Evaluation
class InferenceConfig(GRADEConfig):
#class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    USE_MINI_MASK = False

inference_config = InferenceConfig()
inference_config.DETECTION_NMS_THRESHOLD = 0.5
inference_config.RPN_NMS_THRESHOLD = 0.5
inference_config.display()

# Local path to trained weights file
#COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
#if not os.path.exists(COCO_MODEL_PATH):
#    utils.download_trained_weights(COCO_MODEL_PATH)

model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

#model_path = COCO_MODEL_PATH
model_path = model.find_last()

print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)


dataset_test = GRADEDataset()
dataset_test.load_shapes('/home/ebonetto/GRADE_DATASET/train/images', '/home/ebonetto/GRADE_DATASET/train/masks')
#dataset_test.load_shapes('/home/ebonetto/coco_red/train/images', '/home/ebonetto/coco_red/train/masks')
dataset_test.prepare()
#import ipdb; ipdb.set_trace()
#dataset_test = CocoDataset()
#val_type = "minival"
#coco = dataset_test.load_coco("coco", val_type, return_coco=True, auto_download=True)
#dataset_test.prepare()

image_ids = dataset_test.image_ids


class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']
class_names = ['BG','person']
APs = []
APs0595 = []
vis = True
for image_id in image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_test, inference_config,
                               image_id)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    # Compute AP
    AP, precisions, recalls, overlaps =\
        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'])
    if vis:
        import ipdb; ipdb.set_trace()
        visualize.display_instances(image, r["rois"], r["masks"], r["class_ids"], class_names, r["scores"])
        visualize.display_instances(image, gt_bbox, gt_mask, gt_class_id, class_names)
    APs.append(AP)
    AP =\
        utils.compute_ap_range(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'], verbose=False)
    APs0595.append(AP)
    
print("mAP: ", np.mean(APs))
print("mAP@.5:.95: ", np.mean(APs0595))
