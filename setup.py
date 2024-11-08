# setup
import argparse

from PIL import Image
from icecream import ic
from IPython.display import display, Image as IPImage


# Set up argument parser
parser = argparse.ArgumentParser(description="Script for setting up detectron2 model with Detic.")
parser.add_argument("--vocabulary", type=str, default="lvis", help="Vocabulary to use: lvis, objects365, openimages, or coco")
parser.add_argument("--device", type=str, default="cuda", help="Device to use: cuda or cpu")
parser.add_argument("--seg", type=str, default="instance", help="Segmentation type: instance or panoptic")
args = parser.parse_args()

# Print argument values for confirmation
ic(args.vocabulary, args.device, args.seg)

# ========================

current_dir = str(Path(__file__).parent)+'/'

# Install detectron2
import torch
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)



# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import sys
import numpy as np
import os, json, cv2, random
from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# Detic libraries
sys.path.insert(0, current_dir+'Detic/')
sys.path.insert(0, current_dir+'Detic/datasets/')
sys.path.insert(0, current_dir+'Detic/detic')
sys.path.insert(0, current_dir+'Detic/detic/modeling/utils')
sys.path.insert(0, current_dir+'Detic/third_party/CenterNet2/')
from centernet.config import add_centernet_config
from Detic.detic.config import add_detic_config
from Detic.detic.modeling.utils import reset_cls_test

ic('done here')

folder_path = current_dir+'Detic/'

# Build the detector and download our pretrained weights
cfg = get_cfg()
if args.seg == 'panoptic':
    cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
else:
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(folder_path+"configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml")
    cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True # For better visualization purpose. Set to False for all classes.

if args.device == 'cpu':
    cfg.MODEL.DEVICE='cpu' # uncomment this to use cpu-only mode.

predictor = DefaultPredictor(cfg)

ic('done 2')

# Setup the model's vocabulary using build-in datasets

BUILDIN_CLASSIFIER = {
    'lvis': folder_path+'datasets/metadata/lvis_v1_clip_a+cname.npy',
    'objects365': folder_path+'datasets/metadata/o365_clip_a+cnamefix.npy',
    'openimages': folder_path+'datasets/metadata/oid_clip_a+cname.npy',
    'coco': folder_path+'datasets/metadata/coco_clip_a+cname.npy',
}

BUILDIN_METADATA_PATH = {
    'lvis': 'lvis_v1_val',
    'objects365': 'objects365_v2_val',
    'openimages': 'oid_val_expanded',
    'coco': 'coco_2017_val',
}

# vocabulary = 'lvis' # change to 'lvis', 'objects365', 'openimages', or 'coco'
vocabulary = args.vocabulary if args.vocabulary in BUILDIN_CLASSIFIER else 'lvis'
metadata = MetadataCatalog.get(BUILDIN_METADATA_PATH[vocabulary])
classifier = BUILDIN_CLASSIFIER[vocabulary]
num_classes = len(metadata.thing_classes)
reset_cls_test(predictor.model, classifier, num_classes)
