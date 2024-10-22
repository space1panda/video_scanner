from ultralytics import YOLO
import torch
import cv2
import os
import torch
import sys
from PIL import Image
import numpy as np
import onnxruntime as ort
from torchvision import ops


model = YOLO('yolov8n.pt')
print(model)
print(type(model))


session = ort.InferenceSession("yolov8n.onnx")


def preprocess(im, device='cuda'):
    """
    Prepares input image before inference.

    Args:
        im (torch.Tensor | List(np.ndarray)): BCHW for tensor, [(HWC) x B] for list.
    """
    not_tensor = not isinstance(im, torch.Tensor)
    if not_tensor:
        im = np.stack([cv2.resize(i, (640, 640)) for i in im])
        im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im)

    im = im.to(device)
    im = im.float()  # uint8 to fp16/32a
    if not_tensor:
        im /= 255  # 0 - 255 to 0.0 - 1.0
    return im


def postprocess(self, preds, img, orig_imgs):
    """Post-processes predictions and returns a list of Results objects."""
    preds = ops.non_max_suppression(
        preds,
        self.args.conf,
        self.args.iou,
        agnostic=self.args.agnostic_nms,
        max_det=self.args.max_det,
        classes=self.args.classes,
    )

    if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
        orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

    results = []
    for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0]):
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
    return results


with torch.no_grad():
    res = model.predict('apples.jpg')
    im_orig = [cv2.imread('apples.jpg')]
    im = preprocess(im_orig)
    out = session.run(['output0'], {'images': im.cpu().numpy()})
    out = torch.tensor(out).type_as(im)
    post_process = model.predictor.postprocess(out[0], im, im_orig)
    print('here')


for r in res:
    for data in r.boxes:
        cls_id = data.cls
        score = data.conf
        box = data.xyxy
        print('here')
