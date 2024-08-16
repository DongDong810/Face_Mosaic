import os, sys, random, cv2, time
import detectron2

import numpy as np
import matplotlib.pyplot as plt

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.engine import DefaultTrainer
from detectron2.utils.logger import setup_logger
setup_logger()

from dataset import get_face_dicts

# Register dataset
# train
train_dir = '/workspace/face_mosaic/data/widerface/train/images'
annot_dir = '/workspace/face_mosaic/data/widerface/annotations/'
DatasetCatalog.register("face_train", lambda p='train': get_face_dicts(train_dir, annot_dir, p))
MetadataCatalog.get("face_train").set(thing_classes=["face"])

# val
val_dir = '/workspace/face_mosaic/data/widerface/val/images'
annot_dir = '/workspace/face_mosaic/data/widerface/annotations/'
DatasetCatalog.register("face_val", lambda p='val': get_face_dicts(val_dir, annot_dir, p))
MetadataCatalog.get("face_val").set(thing_classes=["face"])

# Configs
cfg = get_cfg()
backbone = 'mask_rcnn_R_101_FPN_3x'
cfg.merge_from_file(model_zoo.get_config_file(f"COCO-InstanceSegmentation/{backbone}.yaml"))

cfg.DATASETS.TRAIN = ("face_train",)
cfg.DATASETS.TEST = ("face_val",)
cfg.DATALOADER.NUM_WORKERS = 4

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f"COCO-InstanceSegmentation/{backbone}.yaml")  # Let training initialize from model zoo
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # RoIHead batch size (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (face)

cfg.SOLVER.IMS_PER_BATCH = 8  # images per batch (4)
cfg.SOLVER.BASE_LR = 0.00025  # learning rate
cfg.SOLVER.MAX_ITER = 3000    # max iteration (1500)
# cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.STEPS = []         # the checkpoints (number of iterations) at which the learning rate will be reduced by GAMMA
# cfg.SOLVER.GAMMA = 0.05     # learning rate decay

cfg.TEST.EVAL_PERIOD = 500

current_time = time.strftime("%Y%m%d_%H%M")
cfg.OUTPUT_DIR = os.path.join('/workspace/face_mosaic/result', f"{current_time}_{backbone}")

# Train
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
