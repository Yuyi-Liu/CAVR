import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from rearrange_on_proc.arguments import args
from seg.utils.torch_utils import select_device, smart_inference_mode
from seg.models.common import DetectMultiBackend
from seg.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from seg.utils.augmentations import (Albumentations, augment_hsv, classify_albumentations, classify_transforms, copy_paste,
                                 letterbox, mixup, random_perspective)
from seg.utils.segment.general import process_mask, scale_masks
import torch
import numpy as np
from rearrange_on_proc.constants import CATEGORY_LIST, ALL_CATEGORY_LIST


class YOLOV7_SegmentationHelper:
    def __init__(self,gpu_id) -> None:
        self.device = select_device(str(gpu_id))
        self.imgsz=(640, 640)# inference size (height, width)
        weights = f"{os.path.dirname(os.path.realpath(__file__))}/best.pt"
        self.model = DetectMultiBackend(weights, device=self.device, fp16=False)
        self.conf_thres=0.25  # confidence threshold
        self.iou_thres=0.45  # NMS IOU threshold
        self.max_det=1000  # maximum detections per image
        self.classes = None
        self.agnostic_nms=False

    def seg_pred(self,image):
        stride, names, pt = self.model.stride, self.model.names, self.model.pt
        imgsz = check_img_size(self.imgsz, s=stride)  # check image size
        self.model.warmup(imgsz=(1 , 3, *imgsz))
        # im0s = image
        im = letterbox(im0s,  stride=32)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous

        im = torch.from_numpy(im).to(self.device)
        im = im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        result,feature_map_list = self.model(im, augment=False, visualize=False)

        pred,out = result
        proto = out[1]
        pred = non_max_suppression(pred, max_det=self.max_det, nm=32)
    
        im0 = im0s.copy()
        det = pred[0]
        segment_list = []
        if len(det):
            masks = process_mask(proto[0], det[:, 6:], det[:, :4], im.shape[2:], upsample=True) 

            for i in range(len(det)):
                segment_dict = {}

                bbox = det[i][:4]
                # feature_vectors = []
                # for feature_map in feature_map_list:
                #     scale = im.shape[2] / feature_map.shape[2]
                #     scaled_bbox = torch.round(bbox / scale).long()
                #     x_min,y_min,x_max,y_max = scaled_bbox
                #     if x_max == x_min:
                #         x_max += 1
                #     if y_max == y_min:
                #         y_max += 1
                #     feature_area = feature_map[:,:,y_min:y_max,x_min:x_max]
                #     if np.isnan(feature_area.cpu()).sum():
                #         print('yolo seg NaN!!!')
                #     feature_vector = torch.mean(feature_area, dim=(2, 3))
                #     if np.isnan(feature_vector.cpu()).sum():
                #         print('yolo feature NaN!!!')
                #     feature_vectors.append(feature_vector)
                # concat_feature = torch.cat(feature_vectors, dim=1)
                label = det[i][5]
                label = ALL_CATEGORY_LIST[int(label)-1]
                mask_np = masks[i].cpu().numpy()
                mask_np = mask_np.reshape(mask_np.shape+(1,))
                mask_origin_sz = np.squeeze(scale_masks(im.shape[2:], mask_np, im0.shape))
                segment_dict["label"] = label
                segment_dict["mask"] = mask_origin_sz
                segment_dict["feature"] = None
                segment_dict["score"] = det[i][4].item()
                segment_list.append(segment_dict)
        return segment_list

if __name__ == '__main__':
    seg = YOLOV7_SegmentationHelper()
    image = cv2.imread("/home/lyy/rearrange_on_ProcTHOR/rgb.jpg")
    segment_list = seg.seg_pred(image)
    print(segment_list)
    
    