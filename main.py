import cv2
import torch
from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate
import GroundingDINO.groundingdino.datasets.transforms as T
from PIL import Image

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

TEXT_PROMPT = "water bottle"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

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

cap = cv2.VideoCapture(1)
img_h, img_w = None, None
target_object = input("enter the description of the target: ").strip().lower()
frame_count = 0
# target_parameters = [None, None, None]     # boxes, logits, phrases 
trackers = [] #for csrt tracker objects

target = None
while True:
    ret, frame = cap.read()
    if not ret: break

    img_h, img_w = frame.shape[:2]
    # selection_type = auto_or_manual()
    
    if frame_count == 0 : 
        # target_parameters = detect(frame, target_object) 
        # target = show_targets(target_parameters[0], target_parameters[1], target_parameters[2], target_object)

        boxes, logits, phrases = detect(frame, target_object)
        candidates = show_targets(boxes, logits, phrases, target_object)
        trackers = []
        for box, score in candidates:
            x1, y1, x2, y2 = box_to_pixels(box, img_w, img_h)
            x1, y1 = max(x1, 0), max(y1, 0)
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0: continue

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

    # if target:
    #     for box, score in target:
    #         x1, y1, x2, y2 = box_to_pixels(box, img_w, img_h)
    #         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    #         label = f"{target_object} {score:.2f}"
    #         cv2.putText(frame, target_object, (x1, max(y1 -10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()