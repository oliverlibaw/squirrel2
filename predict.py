# Ultralytics YOLO 🚀, GPL-3.0 license 

import hydra
import torch
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
import picamera2 

class DetectionPredictor(BasePredictor):

  def __init__(self, cfg):
    self.cfg = cfg
    self.camera = picamera2.Picamera2()
    self.camera.configure(self.camera.preview_configuration(main={"format": 'RGB888', "size": (416, 416)})) 
    self.camera.start_preview()

  def preprocess(self, frame):
    img = torch.from_numpy(frame).to(self.model.device) 
    img = img.half() # FP16
    img /= 255  
    return img

  # Adjusted max det to 4    
  def postprocess(self, preds, img, orig_img): 
    preds = ops.non_max_suppression(preds, self.cfg.conf, self.cfg.iou, agnostic=self.cfg.agnostic_nms, max_det=4) 
    return preds

  def predict(self):
    while True:
      frame = np.array(self.camera.capture_array())
      predictions = self.predict_on_batch(frame)[0]
      if len(predictions) > 0:
        print(predictions[:, 5], predictions[:, 4]) # classes, confidences


@hydra.main(config_path=".", config_name="infer.yaml")  
def predict(cfg):
    cfg.model = cfg.model or "yolov8n.pt" 
    cfg.imgsz = (416, 416) # Reduced resolution
    cfg.source = None # Use camera 
    predictor = DetectionPredictor(cfg)
    predictor.predict()

if __name__ == "__main__":
  predict()
