import logging
import numpy as np
import skimage
import skimage.morphology
import pdb 
from rearrange_on_proc.arguments import args
import rearrange_on_proc.utils.utils as utils
from rearrange_on_proc.constants import CATEGORY_to_ID, INSTANCE_FEATURE_SIMILARITY_THRESHOLD, INSTANCE_IOU_THRESHOLD
from torch_scatter import scatter_max
import itertools
import torch
import matplotlib.pyplot as plt
import tkinter
import matplotlib
import matplotlib.gridspec as gridspec
from collections import Counter
# matplotlib.use('TkAgg')


class Mapper():
    def __init__(self, C, origin, map_size, resolution, stage,max_depth=164, z_bins=[0.05, 3], num_categories=50, loc_on_map_selem=skimage.morphology.disk(2)):
        # Internal coordinate frame is X, Y into the scene, Z up.
        self.sc = 1
        self.C = C
        self.resolution = resolution
        self.max_depth = max_depth
        self.stage = stage

        self.z_bins = z_bins
        map_sz = int(np.ceil((map_size*100)//(resolution*100)))
        self.map_sz = map_sz
        print("MAP SIZE (pixel):", map_sz)

        self.map = np.zeros((map_sz, map_sz, len(self.z_bins)+1), dtype=np.float32)
        self.map_for_view_distance = np.ones((map_sz, map_sz), dtype=np.float32) *  float('inf')

        self.local_map = np.zeros((map_sz, map_sz, len(self.z_bins)+1), dtype=np.float32)
        self.semantic_map = np.zeros((map_sz, map_sz, len(self.z_bins)+1, num_categories+1), dtype=np.float32)
        self.loc_on_map = np.zeros((map_sz, map_sz), dtype=np.float32)
        self.added_obstacles = np.ones((map_sz, map_sz), dtype=bool)

        self.origin_xz = np.array([origin['x'], origin['z']])
        step_pix = int(args.STEP_SIZE / args.map_resolution)
        self.origin_map = np.array([(self.map.shape[0]-1-step_pix)/2, (self.map.shape[0]-1-step_pix)/2], np.float32)
        #self.origin_map = self._optimize_set_map_origin(self.origin_xz, self.resolution)
        # self.objects = {}
        self.loc_on_map_selem = loc_on_map_selem

        self.num_boxes = 0

        self.step = 0

        self.bounds = None  # bounds

        self.local_point_cloud = None


        self.feature_list = []

        self.walkthrough_point_cloud_list = []

    # def _optimize_set_map_origin(self, origin_xz, resolution):
    #     return (origin_xz + 15) / resolution

    def transform_egocentric_worldCoordinate(self, XYZ):
        R = utils.get_r_matrix([0., 0., 1.], angle=self.current_rotation)
        XYZ = np.matmul(XYZ.reshape(-1, 3), R.T).reshape(XYZ.shape)
        XYZ[:, :, 0] = XYZ[:, :, 0] + self.current_position[0] - self.origin_xz[0] + self.origin_map[0] * self.resolution
        XYZ[:, :, 1] = XYZ[:, :, 1] + self.current_position[1] - self.origin_xz[1] + self.origin_map[1] * self.resolution
        return XYZ

    def update_position_on_map(self, position, rotation):
        self.current_position = np.array(
            [position['x'], position['z']], np.float32)
        self.current_rotation = -np.deg2rad(rotation)
        x, y = self.get_position_on_map()
        self.loc_on_map[int(y), int(x)] = 1
        # TODO(saurabhg): Mark location on map

    def get_position_on_map(self):
        return map_position
    
    def add_observation(self,position, rotation, elevation, depth, seg=None, seg_dict=None, object_track_dict=None, add_obs=True, add_seg = False):

        d = depth*1.
        self.max_depth = args.max_depth
        self.min_depth = args.min_depth
        d[d > self.max_depth] = 0
        d[d < self.min_depth] = np.NaN
        
        d = d / self.sc
        self.update_position_on_map(position, rotation)
        
        
        isnan = np.isnan(XYZ3[:, :, 0])

        # Shape: counts [map_size, map_size, n_z_bins], is_valids [H, W], inds [H, W]
        self.map += counts
        self.local_map = counts
        self.local_point_cloud = XYZ3


        XYZ3[isnan] = 0
        # return isnan,XYZ3,None,None

        # # self.local_point_cloud = 480 * 480 * [x,y,z,score,label,feature_num]
        # local_point_cloud = np.zeros((XYZ4.shape[0],XYZ4.shape[1],6))
        # local_point_cloud[:,:,:3] = XYZ4
        # for category,category_instances in seg_dict.items():
        #     category_id = CATEGORY_to_ID[category]
        #     for instance_id, instance_info in category_instances.items():
        #         mask = instance_info["mask"]
        #         local_point_cloud[mask,3] = instance_info["score"]
        #         feature_num = len(self.feature_list)
        #         local_point_cloud[mask,4] = category_id
        #         local_point_cloud[mask,5] = feature_num
        #         self.feature_list.append(instance_info["feature"])
        # local_point_cloud[isnan] = 0
        # local_point_cloud = local_point_cloud.reshape(-1,6)
        # local_point_cloud = local_point_cloud[local_point_cloud[:, 4] != 0]
        
        
        
        # if len(self.global_point_cloud) == 0:
        #     self.global_point_cloud = {(row[0], row[1], row[2]): row for row in local_point_cloud}
        # else:
        #     for row in local_point_cloud:
        #         key = (row[0], row[1], row[2])
        #         if key in self.global_point_cloud:
        #             if row[3] > self.global_point_cloud[key][3]:
        #                 self.global_point_cloud[key] = row
        #         else:
        #             self.global_point_cloud[key] = row
            
        

        # tmp_d = depth*1
        # tmp_d[tmp_d > 10] = 0
        # tmp_d[tmp_d < 0.02] = np.NaN
        # tmp_d = tmp_d / self.sc 
        # tmp_XYZ1 = utils.get_point_cloud_from_z(tmp_d, self.C)
        
        # view_d = tmp_d[:, :, np.newaxis]
        # view_distance_map = np.squeeze(self.transfrom_feature_to_map(feature = view_d, XYZ = tmp_XYZ3.copy(), xy_resolution=self.resolution, z_bins = self.z_bins))
        
        return isnan,XYZ3,None,None

        # # Shape: 
        # # seg [H, W, num_category + 1], XYZ3 [H, W, 3]
        # if add_seg:
        #     local_semantic = self.transfrom_feature_to_map(feature = seg, XYZ = XYZ3, xy_resolution = self.resolution, z_bins = self.z_bins).astype(np.int8)
        #     seg_same = []
        #     seg_different = []
        #     curr_judge_new_and_centriod = {}
        #     for seg_category, seg_instances in seg_dict.items():
        #         seg_category_id = CATEGORY_to_ID[seg_category]
        #         curr_category_semantic_map = local_semantic[:, :, :, seg_category_id].copy()         #[H, W, len(z)+1]
        #         global_category_semantic_map = self.semantic_map[:, :, :, seg_category_id]
                
        #         curr_category_new_instance_count = 0
        #         for seg_instance_id, seg_instance in seg_instances.items():
        #             # if 'CounterTop' in seg_instance_id:
        #             #     print('debug here')
        #             seg_instance_id_int = int(seg_instance_id.split('_')[-1])
        #             curr_instance_semantic_map = curr_category_semantic_map == seg_instance_id_int
        #             curr_instance_feature = seg_instance['feature']

        #             if curr_instance_semantic_map.sum() == 0:
        #                 if args.debug_print:
        #                     print(f'Sth {seg_instance_id} is not considered in the memory')
        #                 continue
        #             intersection = curr_instance_semantic_map & (global_category_semantic_map != 0)
        #             if intersection.sum() > 0:
        #                 # intersection_global_instance_ids = np.unique(global_category_semantic_map[intersection])
        #                 most_possible_gobal_instance_id = int(Counter(global_category_semantic_map[intersection]).most_common()[0][0])
                        
        #                 global_instance_semantic_map = global_category_semantic_map == most_possible_gobal_instance_id
        #                 global_instance_feature = object_track_dict[seg_category][seg_category + '_' + str(most_possible_gobal_instance_id)]['feature']
                        
        #                 feature_similarity = utils.cosine_similarity(curr_instance_feature, global_instance_feature)
        #                 iou = utils.calculate_iou_numpy(curr_instance_semantic_map, global_instance_semantic_map)
        #                 if feature_similarity < INSTANCE_FEATURE_SIMILARITY_THRESHOLD and iou < INSTANCE_IOU_THRESHOLD:
        #                     curr_category_new_instance_count += 1
        #                     seg_different.append((seg_category, seg_instance['score'], feature_similarity, iou, int(np.max(global_category_semantic_map)) + 1))
        #                     local_semantic[curr_instance_semantic_map, seg_category_id] = int(np.max(global_category_semantic_map) + curr_category_new_instance_count)
        #                     loc_on_map = np.array([np.median(x_list) for x_list in np.where(curr_instance_semantic_map)])
        #                     curr_judge_new_and_centriod[seg_instance_id] = loc_on_map #yxz
        #                 else:
        #                     seg_same.append((seg_category, seg_instance['score'],feature_similarity, iou, most_possible_gobal_instance_id))
        #                     local_semantic[curr_instance_semantic_map, seg_category_id]  = most_possible_gobal_instance_id  
        #                     object_track_dict[seg_category][seg_category + '_' + str(most_possible_gobal_instance_id)]['score'] = (object_track_dict[seg_category][seg_category + '_' + str(most_possible_gobal_instance_id)]['score'] + seg_instance['score']) / 2
        #                     object_track_dict[seg_category][seg_category + '_' + str(most_possible_gobal_instance_id)]['feature'] = np.mean([curr_instance_feature, global_instance_feature], axis = 0)
        #                     loc_on_map = np.array([np.median(x_list) for x_list in np.where(curr_instance_semantic_map | global_instance_semantic_map)])
        #                     object_track_dict[seg_category][seg_category + '_' + str(most_possible_gobal_instance_id)]['centroid'] =  loc_on_map
        #                     if seg_instance["simulator_id"] not in object_track_dict[seg_category][seg_category + '_' + str(most_possible_gobal_instance_id)]["simulator_id"]:
        #                         object_track_dict[seg_category][seg_category + '_' + str(most_possible_gobal_instance_id)]["simulator_id"].append(seg_instance["simulator_id"])
        #                 # if np.array_equal(curr_instance_semantic_map, curr_instance_semantic_map * global_instance_semantic_map) or np.array_equal(global_instance_semantic_map, curr_instance_semantic_map * global_instance_semantic_map):
        #                 # elif utils.calculate_iou_numpy(curr_instance_semantic_map, global_instance_semantic_map) < INSTANCE_IOU_THRESHOLD:

        #             else:
        #                 curr_category_new_instance_count += 1
        #                 seg_different.append((seg_category, seg_instance['score'], "-"))
        #                 local_semantic[curr_instance_semantic_map, seg_category_id] = int(np.max(global_category_semantic_map) + curr_category_new_instance_count)
        #                 loc_on_map = np.array([np.median(x_list) for x_list in np.where(curr_instance_semantic_map)])
        #                 curr_judge_new_and_centriod[seg_instance_id] = loc_on_map

        #     if args.debug_print:
        #         print(f'Seg same: {seg_same} \n Seg diff: {seg_different}')
        #     return object_track_dict, curr_judge_new_and_centriod
        
        # return None, None


                                      
    def transfrom_feature_to_map2(self, feature, XYZ, xy_resolution, z_bins):
        '''
        Input: feature [H, W, feature_dims]. For segmentation,  the 'feature_dims' equals 'num_category+1'
               XYZ [H, W, 3], where '3' represents the coordinates X, Y, Z in the map
        Output: local_feature_map [map_size, map_size, n_z_bins, feature_dims]
        '''
        H,W,xyz_dim = XYZ.shape
        feature_dims = feature.shape[-1]
        feature = feature.reshape(-1, feature_dims).astype(np.float64)
        map_size = self.map.shape[0]

        n_z_bins = len(z_bins) + 1
        local_feature_map = np.zeros((self.map.shape[0], self.map.shape[1], n_z_bins, feature_dims))
        local_feature_map = torch.from_numpy(local_feature_map).view(-1, feature_dims)   #[map_sz*map_sz*bins, feature_dims]

        isnotnan = np.logical_not(np.isnan(XYZ[:, :, 0]))
        X_bin = np.round(XYZ[:, :, 0] / xy_resolution).astype(np.int32)
        Y_bin = np.round(XYZ[:, :, 1] / xy_resolution).astype(np.int32)
        Z_bin = np.digitize(XYZ[:, :, 2], bins=z_bins).astype(

        isvalid = np.array([X_bin >= 0, X_bin < map_size, Y_bin >= 0, Y_bin < map_size,
                            Z_bin >= 0, Z_bin < n_z_bins, isnotnan])  # H*W*7
        valid_index = np.all(isvalid, axis=0).ravel()  # [H*W, ]

        map_index = (Y_bin * map_size + X_bin) * n_z_bins + Z_bin
        map_index = map_index.ravel()  #[H*W, 1]
        map_index = map_index[:, np.newaxis].astype(np.int64)

        map_index_broadcast = np.broadcast_to(map_index, (map_index.shape[0], feature_dims))
        map_index_broadcast_tensor = torch.tensor(map_index_broadcast)
        src_tensor = torch.tensor(feature[valid_index])
        out = torch.zeros_like(local_feature_map)
        out, _ = scatter_max(src_tensor, index = map_index_broadcast_tensor, out = out, dim = 0)
        local_feature_map, _ = torch.max(torch.stack((local_feature_map, out)), dim = 0)
        local_feature_map = torch.round(local_feature_map)

        local_feature_map = local_feature_map.reshape(self.map.shape[0], self.map.shape[1], n_z_bins, feature_dims).numpy().astype(np.int8)


        return local_feature_map
    

    

    def transfrom_feature_to_map(self, feature, XYZ, xy_resolution, z_bins):
        '''
        Input: feature [H, W, feature_dims]. For segmentation,  the 'feature_dims' equals 'num_category+1'
               XYZ [H, W, 3], where '3' represents the coordinates X, Y, Z in the map
        Output: local_feature_map [map_size, map_size, n_z_bins, feature_dims]
        '''
        position_yxz = []
        weight_yxz = []

        feature_dims = feature.shape[-1]
        feature = feature.reshape(-1, feature_dims)
        XYZ[:,:,[0,1]] = XYZ[:,:,[1,0]] # change to (Y, X, Z) for index calculation
        YXZ = XYZ
        H,W,yxz_dims = YXZ.shape
        YXZ = YXZ.reshape(-1, yxz_dims)
        isnotnan = np.logical_not(np.isnan(YXZ[:, 0]))

        map_shape = list(self.map.shape) #[H,W,len(z_bins)]
        map_shape[0], map_shape[1] = map_shape[1], map_shape[0]  # y,x,z
        n_z_bins = len(z_bins) + 1
        local_feature_map = np.zeros((self.map.shape[0], self.map.shape[1], n_z_bins, feature_dims))
        local_feature_map = torch.from_numpy(local_feature_map).view(-1, feature_dims)

        for somedim in range(yxz_dims - 1):
            position_yxSomedim = YXZ[:, somedim] / xy_resolution
            position_nearInt = []
            weight_nearInt = []
            
            for dx in [0, 1]:
                position_ix = np.floor(position_yxSomedim) + dx 
                isvalid = np.array([position_ix >= 0, position_ix < map_shape[somedim], isnotnan])
                # pdb.set_trace()
                isvalid = np.all(isvalid, axis=0)
                isvalid = isvalid.astype(position_yxSomedim.dtype)

                # pdb.set_trace()
                weight_ix = 1 - np.abs(position_yxSomedim - position_ix)
                weight_ix = weight_ix * isvalid
                position_ix = position_ix * isvalid

                # weight_ix[np.logical_not(isvalid)] = 0
                # position_ix[np.logical_not(isvalid)] = 0

                position_nearInt.append(position_ix)
                weight_nearInt.append(weight_ix)
                # pdb.set_trace()
            
            position_yxz.append(position_nearInt)
            weight_yxz.append(weight_nearInt)
        
        Z = np.digitize(YXZ[:, 2], bins = z_bins).astype(np.int32)
        position_yxz.append([Z])
        weight_yxz.append(np.ones_like(Z))

        list_dx = [[0, 1], [0, 1], [0]]
            weight = np.ones_like(weight_yxz[0][0])
            index = np.zeros_like(weight_yxz[0][0])
                index = index * map_shape[somedim] + position_yxz[somedim][dy_dx_dz[somedim]] 

            valid_index = np.logical_not(np.isnan(index))
            index_filterNaN = index[valid_index, np.newaxis]
            index_filterNaN_shape = index_filterNaN.shape[0]
            index_filterNaN = index_filterNaN.astype(np.int64)

            index_broadcast = np.broadcast_to(index_filterNaN, (index_filterNaN_shape,feature_dims))
            index_broadcast_tensor = torch.tensor(index_broadcast)
            src_tensor = torch.from_numpy(feature[valid_index] * weight[valid_index, np.newaxis])
            # local_feature_map.scatter_add_(0, index_broadcast_tensor, src_tensor)
            out = torch.zeros_like(local_feature_map)
            out, _ = scatter_max(src_tensor, index=index_broadcast_tensor, out = out, dim=0)
            # local_feature_map = torch.round(local_feature_map)
        
        local_feature_map = local_feature_map.reshape(self.map.shape[0], self.map.shape[1], n_z_bins, feature_dims).numpy()
        return local_feature_map
            

    def get_rotation_on_map(self):
        map_rotation = self.current_rotation
        return map_rotation

    def add_obstacle_in_front_of_agent(self,act_name,rotation,size_obstacle=10, pad_width=0,held_mode=False):
        '''
        salem: dilation structure normally used to dilate the map for path planning
        '''
        act_rotation = rotation
        if act_name == 'MoveAhead':
            if held_mode:
                return
        elif act_name == 'MoveLeft':
            act_rotation -= args.DT
        elif act_name == 'MoveRight':
            act_rotation += args.DT
        elif act_name == 'MoveBack':
            act_rotation -= 2*args.DT
        act_rotation = -np.deg2rad(act_rotation)

        size_obstacle = self.loc_on_map_selem.shape[0]  # - erosion_size
        loc_on_map_salem_size = int(np.floor(self.loc_on_map_selem.shape[0]/2))
        


        x, y = self.get_position_on_map()
        # print(self.current_rotation)
        if -np.deg2rad(0) == act_rotation:

            ys = [int(y+loc_on_map_salem_size+1),
                  int(y+loc_on_map_salem_size+1+size_obstacle)]
            y_begin = min(ys)
            y_end = max(ys)
            xs = [int(x-np.floor(size_obstacle/2))-pad_width,
                  int(x+np.floor(size_obstacle/2))+pad_width]
            x_begin = min(xs)
            x_end = max(xs)
        elif -np.deg2rad(90) == act_rotation:
            xs = [int(x+loc_on_map_salem_size+1),
                  int(x+loc_on_map_salem_size+1+size_obstacle)]
            x_begin = min(xs)
            x_end = max(xs)
            ys = [int(y-np.floor(size_obstacle/2))-pad_width,
                  int(y+np.floor(size_obstacle/2))+pad_width]
            y_begin = min(ys)
            y_end = max(ys)
        elif -np.deg2rad(180) == act_rotation:
            ys = [int(y-loc_on_map_salem_size-1),
                  int(y-loc_on_map_salem_size-1-size_obstacle)]
            y_begin = min(ys)
            y_end = max(ys)
            xs = [int(x-np.floor(size_obstacle/2))-pad_width,
                  int(x+np.floor(size_obstacle/2))+pad_width]
            x_begin = min(xs)
            x_end = max(xs)
        elif -np.deg2rad(270) == act_rotation:
            xs = [int(x-loc_on_map_salem_size-1),
                  int(x-loc_on_map_salem_size-1-size_obstacle)]
            x_begin = min(xs)
            x_end = max(xs)
            ys = [int(y-np.floor(size_obstacle/2))-pad_width,
                  int(y+np.floor(size_obstacle/2))+pad_width]
            y_begin = min(ys)
            y_end = max(ys)
        else:
            return
            st()
            assert(False)
        self.added_obstacles[y_begin:y_end, x_begin:x_end] = False

    
    def get_obstacle(self):
        obstacle = np.sum(self.map[:, :, 1:-3], 2) >= 10
        return obstacle

    def get_visited(self):
        return self.loc_on_map.astype(np.bool8)

    def get_traversible_map(self, selem, point_count, loc_on_map_traversible):

        obstacle = self.get_obstacle()
        obstacle = skimage.morphology.binary_dilation(obstacle, selem) == True

        traversible = obstacle != True

        # also add in obstacles
        traversible = np.logical_and(self.added_obstacles, traversible)

        if loc_on_map_traversible:

            # traversible_locs = skimage.morphology.binary_dilation(self.loc_on_map, self.loc_on_map_selem) == True
            traversible_locs = self.loc_on_map == True

        if self.bounds is not None:
            # limit to scene boundaries
            bounds_x = [self.bounds[0], self.bounds[1]]
            bounds_z = [self.bounds[2], self.bounds[3]]
            len_x_map = int((max(bounds_x) - min(bounds_x))/self.resolution)
            len_z_map = int((max(bounds_z) - min(bounds_z))/self.resolution)
            half_x_map = len_x_map//2
            half_z_map = len_z_map//2
            x_range = [int(self.origin_map[0]-half_x_map),
                       int(self.origin_map[0]+half_x_map)]
            z_range = [int(self.origin_map[1]-half_z_map),
                       int(self.origin_map[1]+half_z_map)]

            traversible[:z_range[0], :] = False
            traversible[z_range[1]:, :] = False
            traversible[:, :x_range[0]] = False
            traversible[:, x_range[1]:] = False

        return traversible

    # lwj:
    def get_local_map_sum(self):
        local_map_sum = np.sum(self.local_map[:, :, 1:-1], 2)
        local_map_sum[local_map_sum > 100] = 100
        return local_map_sum

    def get_map_for_view_distance(self):
        map_for_view_distance = self.map_for_view_distance.copy()
        map_for_view_distance[np.isinf(map_for_view_distance)] = np.nan
        # map_for_view_distance = map_for_view_distance / self.max_depth
        return map_for_view_distance

    # lwj: 
    def get_explored_map(self, selem, point_count):
        # traversible = self.get_traversible_map(selem, point_count, loc_on_map_traversible=True)
        explored = np.logical_or(explored, traversible)
        if self.bounds is not None:
            # limit to scene boundaries
            bounds_x = [self.bounds[0], self.bounds[1]]
            bounds_z = [self.bounds[2], self.bounds[3]]
            len_x_map = int((max(bounds_x) - min(bounds_x))/self.resolution)
            len_z_map = int((max(bounds_z) - min(bounds_z))/self.resolution)
            half_x_map = len_x_map//2
            half_z_map = len_z_map//2
            x_range = [int(self.origin_map[0]-half_x_map),
                       int(self.origin_map[0]+half_x_map)]
            z_range = [int(self.origin_map[1]-half_z_map),
                       int(self.origin_map[1]+half_z_map)]

            explored[:z_range[0], :] = True
            explored[z_range[1]:, :] = True
            explored[:, :x_range[0]] = True
            explored[:, x_range[1]:] = True
        return explored

    def get_semantic_map_with_maxConfidence_category(self):
        # transfer the 4D semantic map [H, W, len(z)+1, num_category+1]
        # to 3D semantic map with the max confidence category [H, W, len(z)+1]
        shape = self.semantic_map.shape
        semantic_map = self.semantic_map.reshape(-1, shape[-1])
        
        maxConfidence_indices = np.argmax(semantic_map, axis=1)
        maxConfidence_semantic_map = maxConfidence_indices.reshape(shape[:-1]) # [H, W, len(z)+1]
        return maxConfidence_semantic_map

    def get_semantic_map_with_occupy_category(self):
        # transfer the 4D semantic map [H, W, len(z)+1, num_category+1]
        # to 3D semantic map with the max confidence category [H, W, len(z)+1] 
        shape = self.semantic_map.shape
        semantic_map = self.semantic_map.reshape(-1, shape[-1])
        
        occupy_indices = np.ma.masked_equal(semantic_map, 0).argmin(axis = 1)
        occupy_semantic_map = occupy_indices.reshape(shape[:-1])
        return occupy_semantic_map



    def get_topdown_semantic_map(self):
        # transfer the 4D semantic map to Top-Down 2D semantic map
        # Only for further visualization
        '''
        Input: semantic_map [H, W, len(z) + 1, num_category+1], both H, W refer to the map_size(H/W)
        Note: the category in the higher z_bin will be represented
        '''
        # maxConfidence_semantic_map = self.get_semantic_map_with_maxConfidence_category() #[H, W, len(z)+1]
        maxConfidence_semantic_map = self.get_semantic_map_with_occupy_category()
        shape = maxConfidence_semantic_map.shape
        maxConfidence_semantic_map = maxConfidence_semantic_map.reshape(-1, shape[-1]) # [H*W, len(z)+1]

        z_bins_num = maxConfidence_semantic_map.shape[-1] # 0: no category, 1+: object category
        maxConfidence_semantic_map_flip = np.fliplr(maxConfidence_semantic_map)  # to get the highest z_bin to the first index
        z_bin_with_category = maxConfidence_semantic_map_flip > 0 # get the first non-Zero z_bin (i.e. with category)
        first_z_bin_with_category_index = np.argmax(z_bin_with_category, axis=-1) # [H*W, 1]
        first_z_bin_with_category_index = z_bins_num - 1 - first_z_bin_with_category_index
        no_category = ~np.any(maxConfidence_semantic_map, axis=-1)
        first_z_bin_with_category_index[no_category] = 0

        top_down_category_result = maxConfidence_semantic_map[np.arange(maxConfidence_semantic_map.shape[0]), first_z_bin_with_category_index.ravel()]
        top_down_category_result = top_down_category_result.reshape(shape[0], shape[1])
        
        return top_down_category_result

    def get_hierarchical_semantic_map(self, category_id):
        '''
            Input: category
            Output: the semantic_map of that category
            i.e., transfer the 4D semantic map [H, W, len(z)+1, num_category+1] 
                    To 2D category-wise semantic map [H, W] (with counts)
        '''
        # maxConfidence_semantic_map = self.get_semantic_map_with_maxConfidence_category() #[H, W, len(z)+1]
        category_semantic_map = self.semantic_map[:, :, :, category_id]         #[H, W, len(z)+1]
        category_semantic_map = np.amax(category_semantic_map, axis=-1)     #[H, W]
        return category_semantic_map

    def get_topdown_highest_obstacle_map(self):
        # transfer the 3D map to Top-Down 2D highest_obstacle map
        '''
        Input: map [H, W, len(z) + 1], both H, W refer to the map_size(H/W)
        Note: the highest z_bin with obstacle will be represented
        '''
        shape = self.map.shape
        reshaped_map = self.map.reshape(-1, shape[-1])  #[H*W, len(z)+1], each value is the points counts
        z_bins_num = reshaped_map.shape[-1]

        map_flip = np.fliplr(reshaped_map)  # to get the highest z_bin to the first index
        map_flip[:, 0] = 0  # all (>2m, including ceil) are not in consideration
        z_bin_with_point = map_flip > 0  # get the first non-Zero z_bin (i.e. with point counts > 0)
        first_z_bin_with_point_index = np.argmax(z_bin_with_point, axis=-1)  #[H*W, 1]
        first_z_bin_with_point_index = z_bins_num - 1 - first_z_bin_with_point_index
        first_z_bin_with_point_index[no_point] = 0

        first_z_bin_with_point_index = first_z_bin_with_point_index.reshape(shape[0], shape[1])
        return first_z_bin_with_point_index


    def move_map_to_center(self):
        pass
        









    # def process_pickup(self, uuid):
    #     # Upon execution of a successful pickup action, clear out the map at
    #     # the current location, so that traversibility can be updated.
    #     import pdb
    #     pdb.set_trace()

    # def get_object_on_map(self, uuid):
    #     map_channel = 0
    #     if uuid in self.objects.keys():
    #         map_channel = self.objects[uuid]['channel_id']
    #     object_on_map = self.semantic_map[:, :, map_channel]
    #     return object_on_map > np.median(object_on_map[object_on_map > 0])
