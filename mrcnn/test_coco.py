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
from coco import evaluate_coco
from coco import CocoDataset

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


model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

#model_path = COCO_MODEL_PATH
model_path = model.find_last()
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

dataset_val = CocoDataset()
coco = dataset_val.load_coco("/home/ebonetto/coco", "val", year=2017, return_coco=True, class_ids=[1], auto_download=False)
dataset_val.prepare()

evaluate_coco(model, dataset_val, coco, "segm", limit=0)

#ct = COCO("/home/ebonetto/annotations/instances_val2017.json")
#imgIds = ct.getImgIds(catIds=[1])
#imgs = ct.loadImgs(imgIds)


#APs = []
#APs0595 = []
#vis = True

#	results = model.detect([image], verbose=0)
#    r = results[0]
    # Compute AP
#    AP, precisions, recalls, overlaps =\
#        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
#                         r["rois"], r["class_ids"], r["scores"], r['masks'])
#    if vis:
#        import ipdb; ipdb.set_trace()
#        visualize.display_instances(image, r["rois"], r["masks"], r["class_ids"], class_names, r["scores"])
#        visualize.display_instances(image, gt_bbox, gt_mask, gt_class_id, class_names)
#    APs.append(AP)
#    AP =\
#        utils.compute_ap_range(gt_bbox, gt_class_id, gt_mask,
#                         r["rois"], r["class_ids"], r["scores"], r['masks'], verbose=False)
#    APs0595.append(AP)
    
#print("mAP: ", np.mean(APs))
#print("mAP@.5:.95: ", np.mean(APs0595))



#for img in range(len(imgIds)):
#	anns = ct.getAnnIds(imgIds=imgIds[img],catIds=[1])
#	anns = ct.loadAnns(anns)
#	info = ct.loadImgs(imgIds[img])[0]

#	masks = np.zeros(shape=(height,width, len(anns)))
#	for ann in range(len(anns)):
#		masks[:,:,ann] += ct.annToMask(anns[ann])*255
#	masks = masks.astype(np.uint8)
#	d = {'mask': masks,'class': ['human']*len(anns)}
#	np.save(f"/home/ebonetto/coco_red/train/masks/{fname}",d)