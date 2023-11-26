import numpy as np
import pdb
from constants import CATEGORY_to_ID, CATEGORY_LIST, INSTANCE_SEG_THRESHOLD,OPENABLE_OBJECTS
from arguments import args
import sys
import os
import cv2
import torch
import torchvision.models 
import torchvision.transforms as transforms
import torchvision.ops as ops
import torch.nn as nn
from PIL import Image
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from Segmentation.segmentation_helper import MASK_RCNN_SegmentationHelper
from yolov7.yolov7_seg import YOLOV7_SegmentationHelper


class SegmentationHelper():
    def __init__(self, controller,gpu_id) -> None:
        self.controller = controller
        self.use_GT_seg = args.use_GT_seg
        self.seg_utils = args.seg_utils
        self.gpu_id = gpu_id

        # self.objId_to_color = controller.last_event.object_id_to_color
        # self.color_to_objId = controller.last_event.color_to_object_id
        if self.seg_utils == 'GT':           
            self.segmentation_helper = None
        elif self.seg_utils == 'MaskRCNN':
            self.segmentation_helper = MASK_RCNN_SegmentationHelper()
        elif self.seg_utils == 'yolov7':
            self.segmentation_helper = YOLOV7_SegmentationHelper(gpu_id=gpu_id)
        
        # import ssl
 

        # ssl._create_default_https_context = ssl._create_unverified_context

        # self.feature_extractor = torchvision.models.resnet18(pretrained=True)
        # self.feature_extractor=nn.Sequential(*list(self.feature_extractor.children())[:-4])
        # self.feature_extractor.eval()
        # self.transform = transforms.Compose([
        #         transforms.Resize(256),
        #         transforms.CenterCrop(224),
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #     ])
        
    def get_seg_pred(self, rgb, isPickOrPutStage = False):
        '''
        Input: RGB
        Output: Segmentation: H * W * (num_category + 1), Segmented_dict:{'scores':, 'categories', 'masks'} 
        '''

        instance_seg = np.zeros((self.controller.height, self.controller.width, args.num_categories+1))
        segmented_dict = {
            'scores': [],
            'categories': [],
            'masks': [],
        }
        # pdb.set_trace()
        if self.use_GT_seg:
            instance_masks = self.controller.last_event.instance_masks

            for objectId, mask in instance_masks.items():
                category = objectId.split('|')[0]
                if category in OPENABLE_OBJECTS:
                    category_id = CATEGORY_to_ID[category]
                    instance_seg[:,:,category_id] = mask.astype('float')

                    segmented_dict['scores'].append(1.0)
                    segmented_dict['categories'].append(category)
                    segmented_dict['masks'].append(mask.astype(bool))
            return instance_seg.astype(np.uint8), segmented_dict
        elif self.seg_utils == 'yolov7':
            segmented_list = self.segmentation_helper.seg_pred(self.controller.last_event.frame)
            for obj in segmented_list:
                if obj['score'] < INSTANCE_SEG_THRESHOLD:
                    continue
                category = obj["label"]
                mask = obj["mask"]
                # if category in OPENABLE_OBJECTS:
                category_id = CATEGORY_to_ID[category]
                instance_seg[:,:,category_id] = mask.astype('float')
                segmented_dict['scores'].append(obj['score'])
                segmented_dict['categories'].append(category)
                segmented_dict['masks'].append(mask.astype(bool))

        
        return instance_seg.astype(np.uint8), segmented_dict
        

def get_color_feature(rgb, bbox):
    hsv = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2HSV)
    x_min, y_min, x_max, y_max = bbox
    roi = hsv[y_min:y_max, x_min:x_max]
    hist, bins = np.histogram(roi[:,:,0], bins=180, range=[0, 180])
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    # print(hist.shape)
    return torch.tensor(hist)

import cv2
import numpy as np

def visualize_segmentation(image, segmentation, labels):
    """
    """
    
    overlay = image.copy()
    
    num_instances = len(labels)
    colors = np.random.randint(0, 255, size=(num_instances, 3))
    
    for i in range(num_instances):
        overlay[segmentation == i + 1] = colors[i]
        
        y, x = np.where(segmentation == i + 1)
        label_position = (x[0], y[0])
        cv2.putText(overlay, labels[i], label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        
    alpha = 0.6
    output = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
    
    return output





