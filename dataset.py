import os, cv2

from detectron2.structures import BoxMode

# Only for train, val dataset
def get_face_dicts(img_dir, annot_dir, phase) -> list[dict]:
    annot_file = os.path.join(annot_dir, f"wider_face_{phase}_bbx_gt.txt")
    with open (annot_file, 'r') as f:
        lines = f.readlines()

    dataset_dicts = []

    i = 0
    idx = 0
    while i < len(lines):
        record = {}

        image_path = lines[i].strip()
        image_full_path = os.path.join(img_dir, image_path)
        height, width = cv2.imread(image_full_path).shape[:2]

        record["file_name"] = image_full_path
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        objs = []

        num_faces = int(lines[i + 1].strip())
        for j in range(num_faces):
            bbox = list(map(int, lines[i + 2 + j].strip().split())) # # even 1 bbox when num_faces = 0 
            x, y, w, h = bbox[:4] # (x, y) : top-left
            xmin = x
            xmax = x + w
            ymin = y
            ymax = y + h

            poly = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
            poly = [p for x in poly for p in x] # flatten list of list

            obj = {
                 "bbox": [x, y, w, h],
                 "bbox_mode": BoxMode.XYWH_ABS,
                 "segmentation": [poly],
                 "category_id": 0,
            }           
            objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)
        i += max(2 + num_faces, 3) # 3 when num_faces = 0 
        idx += 1

    return dataset_dicts