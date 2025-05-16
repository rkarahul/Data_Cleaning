import torch
import numpy as np
import cv2
import json
import base64
from PIL import Image
import io
import time
import torchvision
from collections import Counter
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.responses import JSONResponse
import multiprocessing
import cv2
import numpy as np
import io
from flask import Flask, jsonify,request, send_file
#memory
import gc
import os
from typing import Dict,List
from pydantic import BaseModel
import signal
#print("num_worker :",multiprocessing.cpu_count())

class SharedModel:
    def __init__(self, weights):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.load_model(weights,self.device)
        #self.model = self.load_model(weights)
        self.classes = self.model.names
        print("\n\nDevice Used:", self.device)
    """
    def load_model(self, weights):
        model = torch.hub.load('yolov5', 'custom', path=weights, source='local')
        model = model.to(self.device)
        return model
    """
    # Function to clean RAM & vRAM
    def clean_memory(self):
        gc.collect()
        torch.cuda.empty_cache()

    def load_model(self, weights,device):
        print("loading device : ",device)
        model = torch.hub.load('yolov5', 'custom', path=weights,source='local',device='cuda:0')#'cuda:0'
        self.clean_memory()
        return model
    def xywh2xyxy(self,x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
        y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
        y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
        y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
        return y
    
    def box_iou(self,box1, box2, eps=1e-7):
        # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            box1 (Tensor[N, 4])
            box2 (Tensor[M, 4])
        Returns:
            iou (Tensor[N, M]): the NxM matrix containing the pairwise
                IoU values for every element in boxes1 and boxes2
        """

        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
        inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

        # IoU = inter / (area1 + area2 - inter)
        return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

    def non_max_suppression(self,
        prediction,
        conf_thres=0.4,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nm=0
        ):

        """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

        Returns:
            list of detections, on (n,6) tensor per image [xyxy, conf, cls]
        """

        # Checks
        assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
        assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
        if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
            prediction = prediction[0]  # select only inference output

        device = prediction.device
        print("NMS divice :",device)
        mps = 'mps' in device.type  # Apple MPS
        if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
            prediction = prediction.cpu()
        bs = prediction.shape[0]  # batch size
        nc = prediction.shape[2] - nm - 5  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates

        # Settings
        # min_wh = 2  # (pixels) minimum box width and height
        max_wh = 7680  # (pixels) maximum box width and height
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        time_limit = 0.5 + 0.05 * bs  # seconds to quit after
        redundant = True  # require redundant detections
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS

        t = time.time()
        mi = 5 + nc  # mask start index
        output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence

            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]):
                lb = labels[xi]
                v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
                v[:, :4] = lb[:, 1:5]  # box
                v[:, 4] = 1.0  # conf
                v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
                x = torch.cat((x, v), 0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # Box/Mask
            box = self.xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
            mask = x[:, mi:]  # zero columns if no masks

            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
            else:  # best class only
                conf, j = x[:, 5:mi].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

            # Apply finite constraint
            # if not torch.isfinite(x).all():
            #     x = x[torch.isfinite(x).all(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            i = i[:max_det]  # limit detections
            if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
                # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = self.box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy

            output[xi] = x[i]
            if mps:
                output[xi] = output[xi].to(device)
            if (time.time() - t) > time_limit:
                print(f'WARNING NMS time limit {time_limit:.3f}s exceeded')
                break  # time limit exceeded

        return output

    
    def prepro_image(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        img = torch.from_numpy(img.transpose((2, 0, 1))).float()  # convert to tensor
        img /= 255.0 
        # Run inference
        img = img.unsqueeze(0)  # add batch dimension
        img = img.to(self.device)
        return img

    def score_frame(self, frame): 
        #self.model.to(self.device)
        pro_img = self.prepro_image(frame)
        #frame = [frame]
        # Inference
        results = self.model(pro_img)
        
        det = self.non_max_suppression(results, conf_thres=0.25, iou_thres=0.45)[0]
        label_color = {
            "Defect": (0, 0, 255),
            }
        #class_ids=[]
        detect=[]
        #bound=[]
        for *xyxy, conf, cls in det:
            lab=self.model.names[int(cls)]
            label = f'{lab} {conf:.2f}'
            bgr = label_color.get(lab, (0, 0, 0))
            #class_ids.append(int(cls))
            detect.append([int(xyxy[0]),int(xyxy[1]),int(xyxy[2]),int(xyxy[3])])
            frame = cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), bgr, 2)
            frame = cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, bgr, 2,cv2.LINE_AA) 
        print("detection boundry with lable :",detect)
        status="NG" if len(detect) > 0 else "OK"
        return frame, status

# Load the shared model once
shared_model = SharedModel(weights=r'E:\Lenskart\spects_inspection\code\yolov5\runs\train\exp4\weights\best.pt')

# === FastAPI setup ===
app = FastAPI()

# Request schema
class ImageRequest(BaseModel):
    image: str         # Base64-encoded image
    roi: List[int]     # Region of interest: [x1, y1, x2, y2]

# Decode base64 image to numpy array
def string_to_image(base64_string: str) -> np.ndarray:
    try:
        img_data = base64.b64decode(base64_string)
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        return np.array(img)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")

# Encode numpy image to base64
def encode_image_to_base64(img):
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode('utf-8')

@app.post("/detect")
async def detect_image(req: ImageRequest):
    t1 = time.time()
    
    # Decode and copy image
    image = string_to_image(req.image)
    original_image = image.copy()

    # Validate ROI
    if len(req.roi) != 4:
        raise HTTPException(status_code=400, detail="ROI must be a list of four integers: [x1, y1, x2, y2]")
    
    x1, y1, w, h = req.roi
    x2=x1+w
    y2=y1+h

    # Apply mask
    mask = np.zeros_like(image)
    mask[y1:y2, x1:x2] = image[y1:y2, x1:x2]
    image_for_detection = mask
    image_for_detection=cv2.resize(image_for_detection,(640,640))
    # Run detection
    resultsImg, status = shared_model.score_frame(image_for_detection)

    processed_base64 = encode_image_to_base64(resultsImg)
    t2 = time.time()
    print(f"Time taken: {t2 - t1:.2f} seconds")

    return JSONResponse(content={
        "image": processed_base64,
        "status": status
    })


@app.get("/ServerCheck")
def read_root():
    return {"status": "server is running"}


if __name__ == "__main__":
    uvicorn.run("backend:app", host="127.0.0.1", port=5000, reload=True)
