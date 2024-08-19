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
from omegaconf import OmegaConf

# Configs
cfg = get_cfg()
cfg_default = OmegaConf.load("/home/minwoo/yai/Face_Mosaic/default.yaml") # Change to your own path!
for k, v in cfg_default.items():
    cfg[k] = v
cfg.merge_from_file(model_zoo.get_config_file(f"COCO-InstanceSegmentation/{cfg.backbone}.yaml"))

# Register dataset
# 1) train
DatasetCatalog.register("face_train", lambda p='train': get_face_dicts(cfg.path.train, cfg.path.annot, p))
MetadataCatalog.get("face_train").set(thing_classes=["face"])

# 2) val
DatasetCatalog.register("face_val", lambda p='val': get_face_dicts(cfg.path.val, cfg.path.annot, p))
MetadataCatalog.get("face_val").set(thing_classes=["face"])

# Settings for training
cfg.DATASETS.TRAIN = ("face_train",)
cfg.DATASETS.TEST = ("face_val",)
cfg.DATALOADER.NUM_WORKERS = 4

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f"COCO-InstanceSegmentation/{cfg.backbone}.yaml")  # Let training initialize from model zoo
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # RoIHead batch size (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (face)

cfg.SOLVER.IMS_PER_BATCH = cfg.batch    # images per batch
cfg.SOLVER.BASE_LR = cfg.lr             # learning rate
cfg.SOLVER.MAX_ITER = cfg.iter          # max iteration
# cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.STEPS = []         # the checkpoints (number of iterations) at which the learning rate will be reduced by GAMMA
# cfg.SOLVER.GAMMA = 0.05     # learning rate decay

cfg.TEST.EVAL_PERIOD = 500

current_time = time.strftime("%Y%m%d_%H%M")
cfg.OUTPUT_DIR = os.path.join(cfg.path.output_root, f"{current_time}_{cfg.backbone}")

# Train
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
