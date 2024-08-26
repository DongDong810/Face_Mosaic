import os, random, cv2
import detectron2

import matplotlib.pyplot as plt

from omegaconf import OmegaConf
from dataset import get_face_dicts
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode

IMAGE_PATH = "/workspace/face_mosaic/data/widerface/test/images/0--Parade/0_Parade_marchingband_1_9.jpg"
VIDEO_PATH = "/workspace/face_mosaic/data/video/video1.mp4"
SAVE_IMAGE_PATH = "/workspace/face_mosaic/vis/mosaiced_image.jpg"
SAVE_VIDEO_PATH = "/workspace/face_mosaic/vis/mosaiced_video.mp4"


def mosaic(image, bboxes, rate=0.3):
    mosaiced_img = image.copy()

    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox.int().tolist()
        roi = image[y_min:y_max, x_min:x_max]
        width, height = roi.shape[:2]
        blur_size = (max(2, int(width * rate)), max(2, int(height * rate)))
        roi = cv2.blur(roi, blur_size) # average filter
        mosaiced_img[y_min:y_max, x_min:x_max] = roi
    
    return mosaiced_img

# Configs
cfg = get_cfg()
cfg_default = OmegaConf.load("default.yaml")
for k, v in cfg_default.items():
    cfg[k] = v
cfg.merge_from_file(model_zoo.get_config_file(f"COCO-InstanceSegmentation/{cfg.backbone}.yaml"))

# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
cfg.MODEL.WEIGHTS = os.path.join(cfg.path.load_ckpt, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  # should be same with training
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # should be same with training
predictor = DefaultPredictor(cfg)


# Inference - image
img = cv2.imread(IMAGE_PATH)
outputs = predictor(img)
mosaiced_img = mosaic(img, outputs["instances"].pred_boxes.tensor, rate=0.5)
cv2.imwrite(SAVE_IMAGE_PATH, mosaiced_img)

# Inference - video
def process_video_frames(video_filename, output_path, model):
    cap = cv2.VideoCapture(video_filename)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Path to save the processed video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Forward pass withe the model
        processed_frame = model_inference(frame, model)

        # Save each frame as a video
        out.write(processed_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Mosaic each frame
def model_inference(frame, model):
    output = model(frame)
    mosaiced_frame = mosaic(frame, output["instances"].pred_boxes.tensor, rate=0.5)
    return mosaiced_frame

process_video_frames(VIDEO_PATH, SAVE_VIDEO_PATH, predictor)
