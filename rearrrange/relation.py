import numpy as np
import torch
import torch.nn as nn
from object_tracker import ObjectTrack
from arguments import args
class Relations_CenterOnly():
    def __init__(self, W, H):  
        self.H = H
        self.W = W
        # self.K = K
        self.Softmax = nn.Softmax(dim=0)

    def _closest(self, ref_centroid, centroids):
        dists = np.sqrt(np.sum((centroids - np.expand_dims(ref_centroid, axis=0))**2, axis=1))
        return dists.argmin()


    def _is_supported_by(self, ref_centroid, centroids, ground_plane_h=None):

        # first check supported by floor
        floor_dist = ref_centroid[1] - ground_plane_h
        if floor_dist<0.1:
            return -1 # floor

        # must be below 
        obj_below = (centroids[:,2] - ref_centroid[2]) < 0.03
        dists = np.sqrt(np.sum((centroids[obj_below] - np.expand_dims(ref_centroid, axis=0))**2, axis=1))

        if len(dists)==0:
            return -2

        argmin_dist = dists.argmin()
        argmin_ = np.arange(centroids.shape[0])[obj_below][argmin_dist]

        return argmin_

    def _is_next_to(self, ref_center, center):
        dist = np.sqrt(np.sum((center - ref_center)**2))
        is_relation = dist < 1.2
        return is_relation
    
class Relation_Calculator():
    def __init__(self,pickupable_objs,receptacles) -> None:
        self.pickupable_objs = pickupable_objs
        self.receptacles = receptacles
        self.relations_util = Relations_CenterOnly(args.H, args.W)
        self.relations_executors_pairs = {
            'next-to': self.relations_util._is_next_to,
            'supported-by': self.relations_util._is_supported_by,
            'closest-to': self.relations_util._closest
            }

    def get_relations_pickupable(self, object_tracker:ObjectTrack):
        
        tracker_centroids, tracker_categories, tracker_instance_ids = object_tracker.get_centroids_and_instance_ids()

        obj_rels = {}
        for obj_i in range(len(tracker_centroids)):

            centroid = tracker_centroids[obj_i]
            obj_instance_id = tracker_instance_ids[obj_i]
            obj_category_name = tracker_categories[obj_i]

            if obj_category_name not in self.pickupable_objs:
                continue

            dists = np.sqrt(np.sum((tracker_centroids - np.expand_dims(centroid, axis=0))**2, axis=1))

            # remove centroids directly overlapping
            dist_thresh = dists>0.05 #self.OT_dist_thresh
            tracker_centroids_ = tracker_centroids[dist_thresh]
            tracker_categories_ = list(np.array(tracker_categories)[dist_thresh])
            if not len(tracker_centroids_):
                continue
        
            # keep only centroids of different labels to compare against
            keep = np.array(tracker_categories_)!=obj_category_name
            tracker_centroids_ = tracker_centroids_[keep]
            tracker_categories_ = list(np.array(tracker_categories_)[keep])
            if not len(tracker_centroids_):
                continue

            keep = []
            for l in tracker_categories_:
                if l not in self.pickupable_objs:
                    keep.append(True)
                else:
                    keep.append(False)
            keep = np.array(keep)
            tracker_centroids_ = tracker_centroids_[keep]
            tracker_categories_ = list(np.array(tracker_categories_)[keep])

            if not len(tracker_centroids_):
                continue

            # ignore floor for now
            relations = self.extract_relations_centroids(centroid, obj_category_name, tracker_centroids_, tracker_categories_, floor_height=-100)

            if obj_category_name not in obj_rels:
                obj_rels[obj_category_name] = {}

            obj_rels[obj_category_name][obj_instance_id] = {
                'relations': relations,
                'centroid': centroid
            }
            # print(relations)

        return obj_rels
    
    def extract_relations_centroids(self, centroid_target, label_target, obj_centroids, obj_labels, floor_height): 

        '''Extract relationships of interest from a list of objects'''

        obj_labels_np = np.array(obj_labels.copy())

        ################# Check Relationships #################
        # check pairwise relationships. this loop is order agnostic, since pairwise relationships are mostly invertible
        
        relations = []
        for relation in self.relations_executors_pairs:
            relation_fun = self.relations_executors_pairs[relation]
            if relation=='closest-to' or relation=='supported-by':
                if relation=='supported-by':
                    if label_target in self.receptacles:
                        continue
                    yes_recept = []
                    for obj_label_i in obj_labels:
                        if obj_label_i in self.receptacles:
                            yes_recept.append(True)
                        else:
                            yes_recept.append(False)
                    yes_recept = np.array(yes_recept)
                    
                    obj_centroids_ = obj_centroids[yes_recept]
                    
                    obj_labels_ = list(obj_labels_np[yes_recept])
                    relation_ind = relation_fun(centroid_target, obj_centroids_, ground_plane_h=floor_height)
                    if relation_ind==-2:
                        pass
                    elif relation_ind==-1:
                        relations.append("The {0} is {1} the {2}".format(label_target, relation.replace('-', ' '), 'Floor'))
                        
                    else:
                        relations.append("The {0} is {1} the {2}".format(label_target, relation.replace('-', ' '), obj_labels_[relation_ind]))
                        
                else:
                    relation_ind = relation_fun(centroid_target, obj_centroids)
                    if relation_ind==-2:
                        pass
                    elif relation_ind==-1:
                        relations.append("The {0} is {1} the {2}".format(label_target, relation.replace('-', ' '), 'Floor'))
                    else:
                        relations.append("The {0} is {1} the {2}".format(label_target, relation.replace('-', ' '), obj_labels[relation_ind]))
            else:
                for i in range(len(obj_centroids)):

                    is_relation = relation_fun(centroid_target, obj_centroids[i])
                
                    if is_relation:
                        relations.append("The {0} is {1} the {2}".format(label_target, relation.replace('-', ' '), obj_labels[i]))

        return relations