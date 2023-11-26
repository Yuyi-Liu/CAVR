from mapper import Mapper
from constants import CATEGORY_LIST
import numpy as np
from arguments import args

class ObjectTrack():
    def __init__(self,mapper:Mapper) -> None:
        self.objects_track_dict = {}
        self.mapper = mapper
        self.dist_threshold = 1
        self.id_index = 0
        self.max_depth = args.max_depth
        self.min_depth = args.min_depth

    def update(self,depth,segmented_dict):
        pred_labels = segmented_dict['categories']
        pred_scores = segmented_dict['scores']
        pred_masks = segmented_dict['masks']

        
        for d in range(len(pred_scores)):
            label = pred_labels[d]
            mask = pred_masks[d]
            score = pred_scores[d]
            centroid = self.get_centroid_from_detection_no_controller(mask, depth)
            if centroid is None:
                continue

            locs = []
            IDs_same = []
            for id_ in self.objects_track_dict.keys():
                if self.objects_track_dict[id_]["label"]==label and self.objects_track_dict[id_]["locs"] is not None:
                    locs.append(self.objects_track_dict[id_]["locs"])
                    IDs_same.append(id_)
            if len(locs)>0:
                locs = np.array(locs)
                dists = np.sqrt(np.sum((locs - np.expand_dims(centroid, axis=0))**2, axis=1))
                dists_sort_inds = np.argsort(dists)
                min_dist_ind = dists_sort_inds[0]
            else:
                min_dist = self.dist_threshold+1
            if min_dist < self.dist_threshold:
                same_ind = min_dist_ind
                same_id = IDs_same[same_ind]
                loc_cur = self.objects_track_dict[same_id]['locs']
                score_cur = self.objects_track_dict[same_id]['scores']
                # add one with highest score if they are the same object
                        
                if score>=score_cur:
                    self.objects_track_dict[same_id]['scores'] = score
                    self.objects_track_dict[same_id]['locs'] = centroid
                else:
                    self.objects_track_dict[same_id]['scores'] = score_cur
                    self.objects_track_dict[same_id]['locs'] = loc_cur
            else:
                self.objects_track_dict[self.id_index] = {}
                self.objects_track_dict[self.id_index]['label'] = label
                self.objects_track_dict[self.id_index]['locs'] = centroid
                self.objects_track_dict[self.id_index]['scores'] = score
                self.id_index += 1
                        
            
    def get_centroid_from_detection_no_controller(self,mask,depth,num_valid_thresh=50):
        yv, xv = np.where(mask)
        depth_box = depth[yv, xv]
        if self.max_depth is not None:
            valid_depth = np.logical_and(valid_depth, depth_box < self.max_depth)
        if self.min_depth is not None:
            valid_depth = np.logical_and(valid_depth, depth_box > self.min_depth)
        where_valid = np.where(valid_depth)[0]
        if len(where_valid) < num_valid_thresh:
            return None
        # print("!!!!!!!!!!!!!!!valid_depth",valid_depth.shape,valid_depth)
        depth_box_valid = depth_box[valid_depth]
        # print("!!!!!!!!!!!!!!!!!!!!!depth_box_valid",depth_box_valid.shape,depth_box_valid)
        argmedian_valid = np.argsort(depth_box_valid)[len(depth_box_valid)//2] 
        argmedian = where_valid[argmedian_valid]

        xv_median = xv[argmedian]
        yv_median = yv[argmedian]

        centroid = self.mapper.local_point_cloud[yv_median][xv_median]

        return centroid
    
    def get_centroids_and_labels(self):
        '''
        get centroids and labels in memory
        object_cat: object category string or list of category strings
        '''
        # order by score 
        scores = []
        IDs = []
        # print("@@@@@@@@@@@@@@",self.objects_track_dict)
        
       
        for key in list(self.objects_track_dict.keys()):
            cur_dict = self.objects_track_dict[key]
            scores.append(cur_dict['scores'])
            IDs.append(key)

        scores_argsort = np.argsort(-np.array(scores))
        IDs = np.array(IDs)[scores_argsort]

        # iterate through with highest score first
        centroids = []
        labels = []
        IDs_ = []
        for key in list(IDs):
            cur_dict = self.objects_track_dict[key]
            centroids.append(cur_dict['locs'])
            labels.append(cur_dict['label'])
            IDs_.append(key)

        return centroids, labels