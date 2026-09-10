import cv2
import torch
from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate
import GroundingDINO.groundingdino.datasets.transforms as T
from PIL import Image
import matplotlib.colors as mcolors
import colorsys
import numpy as np
from turret_class import TurretController
import json
from pathlib import Path
import time

import pybullet as p
import pybullet_data

# uncomment this after 1st run
import os
os.environ["HF_HUB_OFFLINE"] = "1"

# to reduce warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="transformers")
warnings.filterwarnings("ignore", module="torch")
warnings.filterwarnings("ignore", module="timm")

model = load_model(
    "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    "weights/groundingdino_swint_ogc.pth"
    )

BOX_THRESHOLD = 0.45
TEXT_THRESHOLD = 0.35

MANUAL_TESTING = True


transform = T.Compose([
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_image_from_array(frame_bgr):
    img_source = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(img_source)
    image_transformed, _ = transform(image_pil, None)
    return img_source, image_transformed

def detect(frame, target_des):
    img_source, image = load_image_from_array(frame)
    boxes, logits, phrases = predict(model=model, image=image, caption=target_des, box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD)

    return boxes, logits, phrases

def box_to_pixels(box, img_w, img_h):
    cx, cy, w, h = box
    x1 = (cx - w/2) * img_w
    y1 = (cy - h/2) * img_h
    x2 = (cx + w/2) * img_w
    y2 = (cy + h/2) * img_h
    return int(x1), int(y1), int(x2), int(y2)

def show_targets(boxes, logits, phrases, target_phrase):
    candidates = [
        (box, score) for box, score, phrase in zip(boxes, logits, phrases)
        if target_phrase in phrase
    ]
    # if not candidates: return None
    # best_boxes, best_scores = max(candidates, key=lambda x: x[1]) # sort
    # return best_boxes, best_scores

    return candidates

def auto_or_manual():
    selection_type = input("press 0 for automatic and 1 for manual: ").strip()
    if selection_type not in ['0', '1']:
        print("************* Invalid Choice, Exiting ***************\n")
        exit()
    return 0 if selection_type == '0' else 1


# ******************* COLOUR FINDER AND MANAGER ****************
COLOUR_LOOKUP = {}
for name, hexcode in mcolors.XKCD_COLORS.items():
    clean_name = name.replace("xkcd:", "")
    COLOUR_LOOKUP[clean_name] = mcolors.hex2color(hexcode)

def extract_colour_word(target_des, colour_lookup=COLOUR_LOOKUP):
    prompt_lower = target_des.lower()
    matches = [name for name in colour_lookup if name in prompt_lower]
    if not matches: return None, target_des

    target_colour = max(matches, key=len)
    target_des = prompt_lower.replace(target_colour, "").strip()
    print(f"colour detected: {target_colour}\ttarget des: {target_des}")
    return target_colour, target_des

def rgb_to_hsv_range(rgb_float, tolerance_h=20, tolerance_s=60, tolerance_v=60):
    r, g, b = rgb_float
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h_cv, s_cv, v_cv = int(h * 179), int(s * 255), int(v * 255)

    is_grayscale = s_cv < 30

    if is_grayscale:
        lower = np.array([0, 0, max(v_cv - 50, 0)])
        upper = np.array([179, 60, min(v_cv + 50, 255)])

    else:
        lower = np.array([max(h_cv - tolerance_h, 0), 40, 40])
        upper = np.array([min(h_cv + tolerance_h, 179), 255, 255])

    return lower, upper

def check_colour_match(frame, hsv_range, box_coords, is_red=False):
    x1, y1, x2, y2 = box_coords
    crop = frame[y1:y2, x1:x2]

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # mask = cv2.inRange(hsv, hsv_range[0], hsv_range[1])

    if is_red:
        lower1 = np.array([0, hsv_range[0][1], hsv_range[0][2]])
        upper1 = np.array([10, hsv_range[1][1], hsv_range[1][2]])
        lower2 = np.array([170, hsv_range[0][1], hsv_range[0][2]])
        upper2 = np.array([179, hsv_range[1][1], hsv_range[1][2]])
        mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    else:
        mask = cv2.inRange(hsv, hsv_range[0], hsv_range[1])

    match_ratio = np.count_nonzero(mask) / mask.size
    return match_ratio if match_ratio > 0.15 else None


# ***************** SERIAL COORDINATE MANAGER **********************
if not MANUAL_TESTING:
    turret = TurretController()

    file_path = Path("calibration.json")
    if file_path.is_file():
        with open(file_path, "r", encoding='utf-8') as file:
            data = json.load(file)
            pan_coeffs = data["pan_coeffs"]
            tilt_coeffs = data["tilt_coeffs"]
    else:
        print("File Not Found!\nPlease calibrate the system!\n")

    def pixel_to_servo_angle(px, py, pan_coeffs, tilt_coeffs):
        pan = pan_coeffs[0]*px + pan_coeffs[1]*py + pan_coeffs[2]
        tilt = tilt_coeffs[0]*px + tilt_coeffs[1]*py + tilt_coeffs[2]
        return pan, tilt


cap = cv2.VideoCapture(1)
img_h, img_w = None, None
target_object = input("enter the description of the target: ").strip().lower()

colour_name, target_object = extract_colour_word(target_object)

frame_count = 0
# target_parameters = [None, None, None]     # boxes, logits, phrases 
trackers = []           #for csrt tracker objects
target = None

while True:
    ret, frame = cap.read()
    if not ret: break

    img_h, img_w = frame.shape[:2]
    # selection_type = auto_or_manual()
    
    if frame_count == 0 : 
        if colour_name:
            rgb = COLOUR_LOOKUP[colour_name]
            hsv_lower, hsv_upper = rgb_to_hsv_range(rgb)
            print(rgb, hsv_upper, hsv_lower)
        
        boxes, logits, phrases = detect(frame, target_object)
        candidates = show_targets(boxes, logits, phrases, target_object)
        trackers = []
        for box, score in candidates:
            x1, y1, x2, y2 = box_to_pixels(box, img_w, img_h)
            x1, y1 = max(x1, 0), max(y1, 0)
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0: continue

            if colour_name:
                colour_match_result = check_colour_match(frame, (hsv_lower, hsv_upper), (x1, y1, x2, y2))
                print(colour_match_result)
                if colour_match_result == None: 
                    continue
            
            tracker = cv2.legacy.TrackerCSRT_create()
            tracker.init(frame, (x1, y1, w, h))
            trackers.append(tracker) 

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, target_object, (x1, max(y1-10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    else:
        active_tracking = []
        for tracker in trackers:
            success, bbox = tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in bbox]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, target_object, (x, max(y-10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                active_tracking.append(tracker)

        trackers = active_tracking

    frame_count = (frame_count + 1) % 10     # detect every 10th frame

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()