import prior
import numpy as np
import math
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import rearrange_on_proc.utils.utils as utils
import skimage
import torch
import logging
import pdb
import json
from allenact.utils.misc_utils import NumpyJSONEncoder

from ai2thor.controller import Controller
from PIL import Image
from argparse import Namespace
from arguments import args
from mapper import Mapper
from planner import FMMPlanner
from segmentation import SegmentationHelper,get_color_feature
import pyvista as pv

from queue import LifoQueue
from scipy.spatial import distance
from visualization import Animation, visualize_segmentationRGB, visualize_topdownSemanticMap

from rearrange_on_proc.utils.utils import visdomImage, pool2d
from rearrange_on_proc.utils.geom import eul2rotm_py
from rearrange_on_proc.utils.utils import round_to_factor, pose_difference_energy, get_max_height_on_line
from constants import  ROTATE_LEFT, ROTATE_RIGHT, MOVE_AHEAD, MOVE_BACK, MOVE_LEFT, MOVE_RIGHT, DONE, LOOK_DOWN, PICKUP, OPEN, CLOSE, PUT, DROP, LOOK_UP
from constants import POINT_COUNT
from object_tracker import ObjectTrack
from numpy import ma
import skfmm
from torch import nn
import torch.nn.functional as functional
import torchvision.models 
import torchvision.transforms as transforms
import torchvision.ops as ops
import torch.nn as nn
import cv2
from plyfile import PlyData, PlyElement
import matplotlib.pyplot as plt
from visualization import drawTopImg



class Explore():
    def __init__(self, controller: Controller, gpu_id:int,vis: Animation, visdom, curr_stage, map_size, step_max, process_id, solution_config=None, walkthrough_rgbs=None,walkthrough_pointclouds=None,walkthrough_pointclouds_isnan=None,walkthrough_actions=None,walkthrough_depths=None,walkthrough_objs_id_to_pose=None,walkthrough_explorer=None) -> None:
        self.init_on = False
        self.process_id = process_id
        self.controller = controller
        self.gpu_id = gpu_id
        self.curr_stage = curr_stage  # 'walkthrough' or 'unshuffle'
        self.walkthrough_objs_id_to_pose = walkthrough_objs_id_to_pose
        self.walkthrough_explorer = walkthrough_explorer
        if self.curr_stage == 'unshuffle':
            assert self.walkthrough_objs_id_to_pose is not None
            curr_energies_dict = self.get_curr_pose_difference_energy(self.curr_stage)
            curr_energies = np.array(list(curr_energies_dict.values()))
            self.last_energy = curr_energies.sum()
            print(f'unshuffle start_energy: {self.last_energy}')

        self.solution_config = solution_config

        print('Max Steps: ', self.step_max)

        self.W = args.W
        self.H = args.H
        self.STEP_SIZE = args.STEP_SIZE
        self.HORIZON_DT = args.HORIZON_DT
        self.DT = args.DT
        self.pix_T_camX = None
        # actions = ['MoveAhead', 'MoveBack', 'MoveLeft', 'MoveRight',
        #            'RotateRight', 'RotateLeft', 'LookUp', 'LookDown', 'Done']
        # self.act_id_to_name = {i: actions[i] for i in range(len(actions))}
        self.act_id_to_name = {
            ROTATE_LEFT: 'RotateLeft',
            ROTATE_RIGHT: 'RotateRight',
            MOVE_AHEAD: 'MoveAhead',
            MOVE_BACK: 'MoveBack',
            MOVE_LEFT: 'MoveLeft',
            MOVE_RIGHT: 'MoveRight',
            LOOK_DOWN: 'LookDown',
            LOOK_UP: 'LookUp',
            PICKUP: 'PickupObject',
            OPEN: 'OpenObject',
            CLOSE: 'CloseObject',
            PUT: 'PutObject',
            DROP: 'DropHandObject',
        }
        self.act_name_to_id = {value:key for key,value in self.act_id_to_name.items()}

        ar = [args.H, args.W]
        vfov = args.fov * np.pi / 180
        focal = ar[1] / (2 * math.tan(vfov / 2))
        fov = abs(2 * math.atan(ar[0] / (2 * focal)) * 180 / np.pi)
        fov, h, w = fov, ar[1], ar[0]
        C = utils.get_camera_matrix(w, h, fov=fov)


        self.position = {'x': 0.0,
                         'y': 1.5759992599487305,  # fixed when standing up
                         'z': 0.0}
        self.head_tilt = round_to_factor(int(

        self.rotation = 0

        self.invert_pitch = True
        self.camX0_T_origin = self.get_camX0_T_camX(get_camX0_T_origin=True)
        self.camX0_T_origin = utils.safe_inverse_single(self.camX0_T_origin)

        self.map_size = map_size
        self.resolution = args.map_resolution
        if (self.curr_stage == "walkthrough" and args.walkthrough_search == "minViewDistance") or (self.curr_stage == "unshuffle" and args.unshuffle_search == "minViewDistance"):
            self.max_depth = 2  # 4. * 255/25.
        else:
            self.max_depth = 10
        self.mapper_dilation = 1
        loc_on_map_size = int(
            np.floor(self.STEP_SIZE / self.resolution / 2))  # +5
        self.loc_on_map_selem = np.ones(
            (loc_on_map_size * 2 + 1, loc_on_map_size * 2 + 1)).astype(bool)
        self.z = [0.05, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1,
                  1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2]
        self.mapper = Mapper(C, self.position, self.map_size, self.resolution,self.curr_stage,
                             max_depth=self.max_depth, z_bins=self.z, num_categories=args.num_categories,
                             loc_on_map_selem=self.loc_on_map_selem)
        
        self.point_goal = None
        self._setup_execution()

        origin_rgb = self.controller.last_event.frame
        origin_depth = self.controller.last_event.depth_frame
        self.rgb = origin_rgb
        self.depth = origin_depth

        if self.solution_config["unshuffle_match"] == "feature_pointcloud_based":
            self.feature_extractor = torchvision.models.resnet18(pretrained=True)
            self.feature_extractor=nn.Sequential(*list(self.feature_extractor.children())[:-4])
            self.feature_extractor.eval()
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            self.feature_list = []
            if self.curr_stage == "walkthrough":
                self.walkthrough_point_cloud_list = []
                self.walkthrough_rgb_list = []
                self.walkthrough_point_cloud_isnan_list = []
                self.walkthrough_depth_list = []
            elif self.curr_stage == "unshuffle":
                self.walkthrough_point_cloud_list = walkthrough_pointclouds
                self.walkthrough_rgb_list = walkthrough_rgbs
                self.walkthrough_point_cloud_isnan_list = walkthrough_pointclouds_isnan
                self.walkthrough_depth_list = walkthrough_depths
        if self.curr_stage == "unshuffle" and self.solution_config["unshuffle_match"] == "feature_pointcloud_based":
            self.different_point_cloud = {"walkthrough":{},
                                          "unshuffle":{}}
        self.global_point_cloud = {}
        self.local_point_cloud = {}
        if self.curr_stage == "walkthrough" and self.solution_config["unshuffle_search"] == "same":
            self.walkthrough_action_list = []
        elif self.curr_stage == "unshuffle" and self.solution_config["unshuffle_search"] == "same":
            self.walkthrough_action_list = walkthrough_actions
        # if args.use_seg:
        self.segHelper = SegmentationHelper(self.controller,self.gpu_id)

        if args.use_offline_ai2thor:
            self.reachable_positions_in_simulator = self.controller.reachable_points
        else:
            self.controller.step("GetReachablePositions")
            assert self.controller.last_event.metadata["lastActionSuccess"]
            self.reachable_positions_in_simulator = self.controller.last_event.metadata[
                "actionReturn"]
        self.visited_xz = {self.agent_location_in_simulator_to_tuple[:2]}
        self.visited_xzr = {self.agent_location_in_simulator_to_tuple[:3]}

        objects_metadata = self._get_allobjs_info_in_simulator(args.use_offline_ai2thor)
        self.seen_pickupable_objects = set(
            self._get_pickupable_objects_Id(objects_metadata, visible_only=True))
        self.seen_openable_objects = set(self._get_openable_not_pickupable_objects_Id(
            objects_metadata, visible_only=True))
        self.total_pickupable_or_openable_objects = len(set(
            self._get_pickupable_or_openable_objects_Id(objects_metadata, visible_only=False)))

        self.acts = iter(())
        self.path = None

        # self.acts_history = []

        self.obstructed_actions = []

        self.vis = vis
        self.visdom = visdom

        assert self.curr_stage == "walkthrough" or self.curr_stage == "unshuffle"
        if self.curr_stage == "walkthrough":
            self.metrics = {
                f"{self.curr_stage}_actions": [],
                f"{self.curr_stage}_action_successes": [],
                f"{self.curr_stage}/reward": 0,
            }
        else:
            self.metrics = {
                f"{self.curr_stage}_actions": [],
                f"{self.curr_stage}_action_successes": [],
                f"{self.curr_stage}/reward": 0,
                f"pickup_obj_ids":[],
                f"pickup_obj_successes":[],
                f"DropObject_snap_ids":[],
                f"DropObject_snap_successes":[],
                f"open_obj_type":[],
            }

        # self.use_object_tracker = self.solution_config['unshuffle_match'] == 'relation_based'
        # if self.use_object_tracker:
        self.object_tracker = ObjectTrack(mapper=self.mapper)

        self.seed = 0

        self.walk_actions = [
            "LookDown",
            "LookDown",
            "RotateLeft",
            "LookUp",
            "LookUp",
            "RotateLeft",
            "LookDown",
            "LookDown",
            "RotateLeft",
            "LookUp",
            "LookUp",
            "RotateLeft",
            "RotateRight",
            "RotateRight",
            "MoveAhead",
            "MoveAhead",
            "MoveAhead",
            "MoveAhead",
            "MoveAhead",
            "MoveAhead",
            "MoveAhead",
            "MoveAhead",
            "MoveAhead",
            "MoveAhead",
            "MoveAhead",
            "MoveAhead",
            "MoveAhead",
            "MoveAhead",
            "MoveAhead",
            "MoveAhead",
            "MoveLeft",
            "RotateLeft",
            "MoveAhead",
            "MoveAhead",
            "RotateLeft",
            "MoveAhead",
            "MoveAhead",
            "RotateRight",
            "MoveAhead",
            "RotateLeft",
            "MoveAhead",
            "RotateRight",
            "MoveAhead",
            "MoveAhead",
            "MoveAhead",
            "MoveAhead",
            "RotateRight",
            "MoveAhead",
            "RotateLeft",
            "MoveAhead",
            "MoveAhead",
            "MoveAhead",
            "MoveAhead",
            "MoveAhead",
            "MoveAhead",
            "MoveAhead",
            "RotateLeft",
            "MoveAhead",
            "MoveAhead",
            "MoveAhead",
            "MoveAhead",
            "MoveAhead",
            "MoveAhead",
            "MoveAhead",
            "MoveAhead",
            "MoveAhead",
            "MoveAhead",
            "MoveAhead",
            "MoveAhead",
            "MoveAhead",
            "RotateRight",
            "RotateLeft",
            "RotateLeft",
            "MoveAhead",
            "MoveAhead",
            "RotateRight",
            "MoveAhead",
            "MoveAhead",
            "MoveAhead",
            "MoveAhead",
            "MoveAhead",
            "RotateLeft",
            "MoveAhead",
            "RotateRight",
            "MoveAhead",
            "MoveLeft",
            "MoveAhead",
            "MoveLeft",
            "RotateLeft",
            "MoveAhead",
            "MoveAhead",
            "RotateRight",
            "MoveAhead",
            "MoveLeft",
            "RotateLeft",
            "MoveAhead",
            "MoveAhead",
            "MoveAhead",
            "MoveAhead",
            "MoveAhead"
        ]

        if (self.curr_stage == "walkthrough" and args.walkthrough_search == "mass") or (self.curr_stage == "unshuffle" and args.unshuffle_search == "mass"):
            self.mass_exploration_modal = nn.Sequential(
                nn.Conv2d(54, 64, 3, padding=1),

                nn.GroupNorm(1, 64),
                nn.ReLU(),

                nn.Conv2d(64, 64, 3, padding=1),

                nn.GroupNorm(1, 64),
                nn.ReLU(),

                nn.Conv2d(64, 64, 3, padding=1),

                nn.GroupNorm(1, 64),
                nn.ReLU(),

                nn.Conv2d(64, 64, 3, padding=1),

                nn.GroupNorm(1, 64),
                nn.ReLU(),

                nn.Conv2d(64, 1, 3, padding=1),
                )

            self.mass_exploration_modal.load_state_dict(torch.load(args.mass_modal_path))
            self.mass_exploration_modal.eval()
            self.mass_exploration_modal = self.mass_exploration_modal.to(torch.device(f"cuda:{args.mass_device}"))


    def get_camX0_T_camX(self, get_camX0_T_origin=False):
        '''
        Get transformation matrix between first position (camX0) and current position (camX)
        '''
        position = np.array(list(self.position.values()))

        # in aithor negative pitch is up - turn this on if need the reverse
        head_tilt = self.head_tilt
        if self.invert_pitch:
            head_tilt = -head_tilt

        rx = np.radians(head_tilt)  # pitch
        rotation = self.rotation
        if rotation >= 180:
            rotation = rotation - 360
        if rotation < -180:
            rotation = 360 + rotation
        ry = np.radians(rotation)  # yaw
        rz = 0.  # roll is always 0
        rotm = eul2rotm_py(np.array([rx]), np.array(
        origin_T_camX = np.eye(4)
        origin_T_camX[0:3, 0:3] = rotm
        origin_T_camX = torch.from_numpy(origin_T_camX)
        if get_camX0_T_origin:
            camX0_T_camX = origin_T_camX
        else:
            camX0_T_camX = torch.matmul(self.camX0_T_origin, origin_T_camX)
        return camX0_T_camX

    def _setup_execution(self):
        self.execution = LifoQueue(maxsize=200)
        fn = lambda: self._cover_fn()
        self.execution.put(fn)
        fn = lambda: self._init_fn()
        self.execution.put(fn)

    def _init_fn(self):
        self.init_on = True
        for i in range(int(360 / self.DT)):
            # if i % 2 == 0:
            #     for j in range(int(60 / self.HORIZON_DT)):
            #         yield LOOK_DOWN
            # else:
            #     for j in range(int(60 / self.HORIZON_DT)):
            #         yield LOOK_UP

            yield ROTATE_LEFT

            self.init_on = False

    def _sample_point_with_mass_network(self):

        occupancy_map = self.mapper.map
        semantic_map = (self.mapper.semantic_map[:,:,:,:54] > 0).astype(int)
        semantic_map[:,:,:,0] = occupancy_map
        semantic_map_tor = torch.from_numpy(semantic_map).to(torch.device(f"cuda:{args.mass_device}"))
        semantic_map_tor = semantic_map_tor.type(torch.cuda.FloatTensor)

        map_size = occupancy_map.shape[0]
        prediction = self.mass_exploration_modal(semantic_map_tor.sum(dim=2).unsqueeze(0).permute(0, 3, 1, 2))
        prediction = functional.softmax(prediction.view(map_size * map_size), dim=0)
        goal = int(torch.multinomial(prediction, 1))
        return goal // map_size, goal % map_size

    def _cover_fn(self) -> None:
        unexplored = self._get_unexplored()
        if args.debug_print:
            print("Unexplored", np.sum(unexplored))
        explored = np.sum(unexplored) < 20
        if explored:
            print('Unexplored area < 20. Exploration finished')
        else:
            if (self.curr_stage == "walkthrough" and args.walkthrough_search == "mass") or (self.curr_stage == "unshuffle" and args.unshuffle_search == "mass"):
                ind_i,ind_j = self._sample_point_with_mass_network()
            else:
                ind_i, ind_j = self._sample_point_in_unexplored_reachable(unexplored)
            
            if ind_i is None or ind_j is None:
                return
            
            self.point_goal = [ind_i, ind_j]
            fn = lambda: self._cover_fn()
            self.execution.put(fn)
            fn = lambda: self._point_goal_fn_assigned(
                np.array([ind_j, ind_i]), explore_mode=True, dist_thresh=0.5)
            self.execution.put(fn)

    def _get_unexplored(self):
        reachable = self._get_reachable_area()
        explored_point_count = 1
        explored = self.mapper.get_explored_map(
        unexplored1 = np.logical_and(unexplored, reachable)
        # added to remove noise effects
        disk = skimage.morphology.disk(2)
        unexplored2 = skimage.morphology.binary_opening(unexplored1, disk)
        self.unexplored_area = np.sum(unexplored2)

        return unexplored2

    def _get_reachable_area(self):
        traversible = self.mapper.get_traversible_map(
        state_xy = self.mapper.get_position_on_map()
        state_xy = state_xy.astype(np.int32)
        if args.walkthrough_search == "cover" or args.walkthrough_search == "minViewDistance" or args.walkthrough_search == 'cover_nearest' or args.walkthrough_search == 'cover_continue':
            step_pix = int(args.STEP_SIZE / args.map_resolution)
            pooled_map = pool2d(traversible,kernel_size=step_pix,stride=step_pix,pool_mode='avg')
            indices = np.indices(traversible.shape) // step_pix
            traversible = pooled_map[indices[0], indices[1]]
            traversible = traversible > 0.8
            index = state_xy - state_xy % step_pix
            traversible[index[1]:index[1]+step_pix,index[0]:index[0]+step_pix] = True
        planner = FMMPlanner(traversible, 360 // self.DT, int(self.STEP_SIZE /
                             self.mapper.resolution), self.obstructed_actions, self.visdom)
    
        state_theta = self.mapper.get_rotation_on_map() + np.pi / 2
        reachable = planner.set_goal(state_xy)  # dd_mask
        return reachable

    def _sample_point_in_unexplored_reachable(self, unexplored):
        # Given the map, sample a point randomly in the open space.est

        if (self.curr_stage == "walkthrough" and (args.walkthrough_search == "cover_nearest" )) or (self.curr_stage == "unshuffle" and (args.unshuffle_search == "cover_nearest")) :
            unexplored_indexs = np.transpose(np.where(unexplored))
            state_xy = self.mapper.get_position_on_map()
            dist = distance.cdist(np.expand_dims(np.array([int(state_xy[1]),int(state_xy[0])]), axis=0), unexplored_indexs)[0]
            dist_meter = dist * args.map_resolution
            without_thresh = dist_meter > 0.75
            dist = dist[without_thresh]
            unexplored_indexs = unexplored_indexs[without_thresh]

            sorted_dist_indices = np.argsort(dist)

            dist = dist[sorted_dist_indices]
            unexplored_sorted_points = unexplored_indexs[sorted_dist_indices]
            if len(unexplored_sorted_points) == 0:
                return None, None
        
            unexplored_sorted_points = unexplored_sorted_points[0:1000]
            dist = dist[0:1000]

            topdown_highest_obstacle_map = self.mapper.get_topdown_highest_obstacle_map()

            unexplored_max_heights = []
            for coords in unexplored_sorted_points:
                max_height = get_max_height_on_line(topdown_highest_obstacle_map, coords, np.array([int(state_xy[1]),int(state_xy[0])]))
                unexplored_max_heights.append(max_height)
            # sorted_max_height_indices = np.lexsort((unexplored_max_heights, np.arange(len(unexplored_max_heights))))
            sorted_max_height_indices = np.lexsort((dist, unexplored_max_heights))

            min_indices = sorted_max_height_indices[:3]
            rng = np.random.RandomState(self.seed)
            self.seed += 1
        
            min_indice = rng.choice(min_indices)
        
            min_height_coord = unexplored_sorted_points[min_indice]
            return min_height_coord[0], min_height_coord[1]

        ind_i, ind_j = np.where(unexplored)
        rng = np.random.RandomState(self.seed)
        self.seed += 1
        ind = rng.randint(ind_i.shape[0])
        return ind_i[ind], ind_j[ind]

    def _point_goal_fn_assigned(self, goal_loc_cell: np.ndarray, explore_mode: bool, dist_thresh: float, held_mode: bool = False, iters=20):
        '''
        print("point_goal_fn")
        state_xy = self.mapper.get_position_on_map()
        state_xy = state_xy.astype(np.int32)
        state_theta = self.mapper.get_rotation_on_map() + np.pi / 2
        reached = self._check_point_goal_reached(goal_loc_cell, dist_thresh)
        if reached:  # or dists_equal:
            if args.debug_print:
                print("REACHED")
            if explore_mode:
                for _ in range(360 // self.DT):
                    yield LOOK_DOWN
                    yield ROTATE_LEFT
                    # yield LOOK_UP
                    # yield LOOK_DOWN
                    # yield LOOK_DOWN
                    # yield LOOK_UP
            return
        else:
            if iters == 0:
                return
            traversible = self.mapper.get_traversible_map(
                self.selem_agent_radius, POINT_COUNT, loc_on_map_traversible=True)
            planner = FMMPlanner(traversible, 360 // self.DT, int(
                self.STEP_SIZE / self.mapper.resolution), self.obstructed_actions, self.visdom)
            goal_loc_cell = goal_loc_cell.astype(np.int32)
            reachable = planner.set_goal(goal_loc_cell)
            if args.debug_print:
                print("goal_loc_cell", goal_loc_cell)
                print("reachable[state_xy[1], state_xy[0]]", reachable[state_xy[1], state_xy[0]])
                # if not reachable[state_xy[1], state_xy[0]]:
                #     print('debug here !!!!!!!!')
            if reachable[state_xy[1], state_xy[0]]:
                # a, state, act_seq = planner.get_action(np.array([state_xy[0], state_xy[1], state_theta]))
                act_seq, path = planner.get_action_sequence_dij(state_xy, self.rotation, goal_loc_cell, held_mode)
            else:
                a, state, act_seq = planner.get_action(np.array([state_xy[0], state_xy[1], state_theta]))
                path = None
            self.act_seq = act_seq
            self.path = path
            if explore_mode:
                for a in act_seq[:10]:
                    yield a
            else:
                for a in act_seq:
                    yield a
            fn = lambda: self._point_goal_fn_assigned(
                goal_loc_cell, explore_mode=explore_mode, dist_thresh=dist_thresh, iters=iters - 1, held_mode=held_mode)
            self.execution.put(fn)

    def _check_point_goal_reached(self, goal_loc_cell, dist_thresh=0.5):
        state_xy = self.mapper.get_position_on_map()
        state_xy = state_xy.astype(np.int32)
        dist = np.sqrt(np.sum(np.square(state_xy - goal_loc_cell))
                       ) * self.mapper.resolution
        topdown_highest_obstacle_map = self.mapper.get_topdown_highest_obstacle_map()

        max_height = get_max_height_on_line(topdown_highest_obstacle_map, np.array(
            [goal_loc_cell[1], goal_loc_cell[0]]), np.array([state_xy[1], state_xy[0]]))

        return dist < dist_thresh and max_height < 16

    def check_successful_action(self, rgb: np.ndarray = None, rgb_prev: np.ndarray = None, perc_diff_thresh: float = None) -> bool:
        return self.controller.last_event.metadata['lastActionSuccess']
        num_diff = np.sum(np.sum(rgb_prev.reshape(
            self.W * self.H, 3) - rgb.reshape(self.W * self.H, 3), 1) > 0)

        if num_diff < perc_diff_thresh * self.W * self.H:
            success = False
        else:
            success = True
        # self.rgb_prev = rgb
        return success

    def _get_agent_location_in_simulator(self):
        metadata = self.controller.last_event.metadata
        return {
            "x": metadata["agent"]["position"]["x"],
            "y": metadata["agent"]["position"]["y"],
            "z": metadata["agent"]["position"]["z"],
            "rotation": metadata["agent"]["rotation"]["y"],
            "horizon": metadata["agent"]["cameraHorizon"],
            "standing": metadata.get("isStanding", metadata["agent"].get("isStanding")),
        }

    # get xzrsh
    @property
    def agent_location_in_simulator_to_tuple(self):
        agent_loc = self._get_agent_location_in_simulator()
        return (
            round(agent_loc["x"], 2),
            round(agent_loc["z"], 2),
            round_to_factor(agent_loc["rotation"], 90) % 360,
            1 * agent_loc["standing"],
            round_to_factor(agent_loc["horizon"], 30) % 360,
        )

    def _get_allobjs_info_in_simulator(self, use_offline_ai2thor):
        if use_offline_ai2thor:
            objects_metadata = self.controller.all_objects()
        else:
            objects_metadata = self.controller.last_event.metadata['objects']
        assert len(objects_metadata) != 0, f"objects metadata == 0, please check!"
        return objects_metadata

    def _get_pickupable_objects_Id(self, objects_metadata, visible_only=False):
        return [
            o['name'] for o in objects_metadata
            if (o['visible'] or not visible_only) and o['pickupable']
        ]

    def _get_openable_not_pickupable_objects_Id(self, objects_metadata, visible_only=False):
        return [
            o['name'] for o in objects_metadata
            if (o['visible'] or not visible_only) and (o['openable'] and not o['pickupable'])
        ]

    def _get_pickupable_or_openable_objects_Id(self, objects_metadata, visible_only=False):
        return [
            o['name'] for o in objects_metadata
            if (o['visible'] or not visible_only) and (o['openable'] or o['pickupable'])
        ]

    def _update_metrics(self, action_name, action_success, paras = dict(),calReward = False):
        # if self.curr_stage == "unshuffle" and self.step_from_stage_start>99:
            # Image.fromarray(self.controller.last_event.frame).save(f"supplementary/rearrange_img/{self.step_from_stage_start}.jpg")
            # top_image = drawTopImg(self.controller)
            # Image.fromarray(top_image).save(f"supplementary/rearrange_top_image/{self.step_from_stage_start}_topImage.jpg")
        
        self.metrics[f"{self.curr_stage}_actions"].append(action_name)
        self.metrics[f"{self.curr_stage}_action_successes"].append(action_success)
        if(action_name == 'PickupObject'):
            self.metrics[f"pickup_obj_ids"].append(paras["objectId"])
            self.metrics[f"pickup_obj_successes"].append(action_success)
        if(action_name == 'DropObject_snap'):
            self.metrics[f"DropObject_snap_ids"].append(paras["objectId"])
            self.metrics[f"DropObject_snap_successes"].append(action_success)
        if(action_name == "OpenObject"):
            self.metrics[f"open_obj_type"].append(paras["object_type"])
        


        if self.curr_stage == 'walkthrough':
            total_seen_before = len(self.seen_pickupable_objects) + len(self.seen_openable_objects)
            prop_seen_before = total_seen_before / self.total_pickupable_or_openable_objects

            # Updating (recorded) visited locations in simulator (only for metrics calculation)
            agent_loc_tuple = self.agent_location_in_simulator_to_tuple
            self.visited_xz.add(agent_loc_tuple[:2])
            self.visited_xzr.add(agent_loc_tuple[:3])

            objects_metadata = self._get_allobjs_info_in_simulator(use_offline_ai2thor=args.use_offline_ai2thor)
            # Updating seen openable
            for objId in self._get_openable_not_pickupable_objects_Id(objects_metadata, visible_only=True):
                if objId not in self.seen_openable_objects:
                    self.seen_openable_objects.add(objId)

            # Updating seen pickupable
            for objId in self._get_pickupable_objects_Id(objects_metadata, visible_only=True):
                if objId not in self.seen_pickupable_objects:
                    self.seen_pickupable_objects.add(objId)

            total_seen_after = len(
                self.seen_pickupable_objects) + len(self.seen_openable_objects)
            prop_seen_after = total_seen_after / self.total_pickupable_or_openable_objects
            if calReward:
                reward = 5 * (prop_seen_after - prop_seen_before)
                if action_name == 'Pass' and prop_seen_after > 0.5:
                    reward += 5 * (prop_seen_after + (prop_seen_after > 0.98))

        elif self.curr_stage == 'unshuffle':
            if calReward:
            # if action_name in ['PickupObject','OpenObject','CloseObject','PutObject','DropObject']:
                curr_energies_dict = self.get_curr_pose_difference_energy(self.curr_stage)
                curr_energies = np.array(list(curr_energies_dict.values()))
                curr_energy = curr_energies.sum()
                # changed_objs_id = [k for k, v in curr_energies_dict.items() if v > 0]
                # changed_objs_parent_goal = [self.walkthrough_objs_id_to_pose[id]['parentReceptacles'] for id in changed_objs_id]
                # changed_objs_parent_curr = [curr_objs_id_to_pose[id]['parentReceptacles'] for id in changed_objs_id]
                # changed = [f"{changed_objs_id[i]}:from {changed_objs_parent_goal[i]} --> {changed_objs_parent_curr[i]}" for i in range(len(changed_objs_id))]

                energy_change = self.last_energy - curr_energy
                self.last_energy = curr_energy
                reward = energy_change
            # else:
            #     reward = 0
        if calReward:
            self.metrics[f"{self.curr_stage}/reward"] += reward
            if args.debug_print or self.step_from_stage_start % 10 == 0:
                print(f'Process{self.process_id}-{self.curr_stage} Step {self.step_from_stage_start}: {action_name}, --(reward: {reward})')
        else:
            if args.debug_print or self.step_from_stage_start % 1 == 0:
            # if args.debug_print or self.step_from_stage_start % 10 == 0:
                print(f'Process{self.process_id}-{self.curr_stage} Step {self.step_from_stage_start}: {action_name}')


    def get_current_objs_id_to_pose(self, stage = None):
        objs_id_to_pose = dict()
        if stage == 'rearrange':
            objects_metadata = self._get_allobjs_info_in_simulator(use_offline_ai2thor=False)
        else:
            objects_metadata = self._get_allobjs_info_in_simulator(use_offline_ai2thor=args.use_offline_ai2thor)
        for obj in objects_metadata:
            if "Cracked" in obj["name"]:
                continue
            if obj['openable'] or obj.get('objectOrientedBoundingBox') is not None:
                if 'Cracked' in obj['name']:
                    continue
                assert obj['name'] not in objs_id_to_pose
                objs_id_to_pose[obj['name']] = obj

        if self.curr_stage == 'unshuffle':
            for objId_walkthrough in self.walkthrough_objs_id_to_pose.keys():
                if objId_walkthrough not in objs_id_to_pose.keys():
                   # assume the disappeared objects are broken
                    objs_id_to_pose[objId_walkthrough] = {
                        **self.walkthrough_objs_id_to_pose[objId_walkthrough],
                        "isBroken": True,
                        'broken': True,
                        "position": None,
                        "rotation": None,
                        "openness": None,
                    }
            assert len(self.walkthrough_objs_id_to_pose.keys()) == len(objs_id_to_pose.keys()), \
                f"obj poses dismatch ! walkthrough - unshuffle = {set(self.walkthrough_objs_id_to_pose.keys() - set(objs_id_to_pose.keys()))}, \
            unshuffle - walkthrough = {set(objs_id_to_pose.keys()) - set(self.walkthrough_objs_id_to_pose.keys())}"

            if self.step_from_stage_start == 0:
                # If we find a broken goal object, we will simply pretend as though it was not
                # broken. This means the agent can never succeed in unshuffling, this means it is
                # possible that even a perfect agent will not succeed for some tasks.
                broken_objs_id_in_walkthrough = [
                    objId for objId, obj in self.walkthrough_objs_id_to_pose.items() if obj['isBroken']]
                for broken_objId in broken_objs_id_in_walkthrough:
                    self.walkthrough_objs_id_to_pose[broken_objId]["isBroken"] = False
                    objs_id_to_pose[broken_objId]["isBroken"] = False

        return objs_id_to_pose

    def update_position_and_rotation(self, act_id: int) -> None:
        '''
        if 'Rotate' in self.act_id_to_name[act_id]:
            if 'Left' in self.act_id_to_name[act_id]:
                self.rotation -= self.DT
            else:
                self.rotation += self.DT
            self.rotation %= 360
        elif 'Move' in self.act_id_to_name[act_id]:
            if act_id == MOVE_AHEAD:
                self.position['x'] += np.sin(self.rotation /
                                             180 * np.pi) * self.STEP_SIZE
                self.position['z'] += np.cos(self.rotation /
                                             180 * np.pi) * self.STEP_SIZE
            elif act_id == MOVE_BACK:
                self.position['x'] -= np.sin(self.rotation /
                                             180 * np.pi) * self.STEP_SIZE
                self.position['z'] -= np.cos(self.rotation /
                                             180 * np.pi) * self.STEP_SIZE
            elif act_id == MOVE_LEFT:
                self.position['x'] -= np.cos(self.rotation /
                                             180 * np.pi) * self.STEP_SIZE
                self.position['z'] += np.sin(self.rotation /
                                             180 * np.pi) * self.STEP_SIZE
            elif act_id == MOVE_RIGHT:
                self.position['x'] += np.cos(self.rotation /
                                             180 * np.pi) * self.STEP_SIZE
                self.position['z'] -= np.sin(self.rotation /
                                             180 * np.pi) * self.STEP_SIZE
        elif 'Look' in self.act_id_to_name[act_id]:
            if 'Down' in self.act_id_to_name[act_id]:
                self.head_tilt += self.HORIZON_DT
            else:
                self.head_tilt -= self.HORIZON_DT

    def reset_execution_and_acts(self, act_id: int) -> None:
        '''
        '''
        self.execution = LifoQueue(maxsize=200)
        self.acts = None
        fn = lambda: self._cover_fn()
        self.execution.put(fn)
        if(self.point_goal == None):
            return
        self.point_goal = self.choose_reachable_map_pos_in_same_room(
            self.point_goal, thresh=0.5, explore_mode=True)
        if self.point_goal is None:
            return 
        ind_i, ind_j = self.point_goal
        fn = lambda: self._point_goal_fn_assigned(goal_loc_cell=np.array(
            [ind_j, ind_i]), dist_thresh=0.5, explore_mode=True)
        self.execution.put(fn)
        fn = lambda: self._random_move_fn(act_id,held_mode=False)
        self.execution.put(fn)

    def _increment_num_steps_taken(self):
        self.step_from_stage_start += 1

    def choose_reachable_map_pos_in_same_room(self, map_pos, thresh=1, within_thresh=False, explore_mode = False):

        if args.debug_print:
            print("choose_reachable_map_pos_in_same_room")
        reachable = self._get_reachable_area()
        step_pix = int(args.STEP_SIZE / args.map_resolution)

        map_pos = [int(map_pos[0]), int(map_pos[1])]
        pooled_reachable = pool2d(reachable, kernel_size=step_pix, stride=step_pix, pool_mode='avg')

        pooled_reachable = pooled_reachable == 1
        original_reachable_indexs = np.transpose(np.where(reachable))
        pooled_reachable_indexs = (original_reachable_indexs / step_pix).astype(int)
        valid_indices = np.logical_and(pooled_reachable_indexs[:, 0] < pooled_reachable.shape[0], pooled_reachable_indexs[:, 1] < pooled_reachable.shape[1])

     
        final_indexs = original_reachable_indexs[valid_indices][pooled_reachable[pooled_reachable_indexs[valid_indices, 0], pooled_reachable_indexs[valid_indices, 1]] == 1]
        # print("final_index.size", final_indexs.shape)
        dist = distance.cdist(np.expand_dims(map_pos, axis=0), final_indexs)[0]
        sorted_dist_indices = np.argsort(dist)

        dist = dist[sorted_dist_indices]
        reachable_sorted_points = final_indexs[sorted_dist_indices]

            dist_outof_thresh = (dist * args.map_resolution > thresh)
            reachable_sorted_points_outof_thresh = reachable_sorted_points[dist_outof_thresh]
        # if explore_mode and len(reachable_sorted_points_outof_thresh):
            reachable_sorted_points = reachable_sorted_points_outof_thresh
            dist = dist[dist_outof_thresh]
        else:
            dist_within_thresh = (dist * args.map_resolution < thresh)
            reachable_sorted_points_within_thresh = reachable_sorted_points[dist_within_thresh]
            reachable_sorted_points = reachable_sorted_points_within_thresh
            dist = dist[dist_within_thresh]

        if not len(reachable_sorted_points):
            return None
        # reachable_sorted_points_2 = reachable_sorted_points[1000:2000]
        reachable_sorted_points = reachable_sorted_points[0:1000]
        dist = dist[0:1000]
        
        topdown_highest_obstacle_map = self.mapper.get_topdown_highest_obstacle_map()

        reachable_max_heights = []
        for coords in reachable_sorted_points:
            max_height = get_max_height_on_line(topdown_highest_obstacle_map, coords, map_pos)
            reachable_max_heights.append(max_height)
        
        # sorted_max_height_indices = np.lexsort((reachable_max_heights, np.arange(len(reachable_max_heights))))
        sorted_max_height_indices = np.lexsort((dist, reachable_max_heights))
        # sorted_max_height_indices = np.argsort(reachable_max_heights)
        min_indices = sorted_max_height_indices[:3]
        rng = np.random.RandomState(self.seed)
        self.seed += 1
        
        if len(min_indices) == 0:
            print('debug here')
        min_indice = rng.choice(min_indices)
        # print("reachable_max_heights", reachable_max_heights)
        # print("rng",rng)
        # print("seed",self.seed)
        # print("min_indices",min_indices)
        # print("min_indice",min_indice)
        
        min_height_coord = reachable_sorted_points[min_indice]
        return min_height_coord[0], min_height_coord[1]

    def _random_move_fn(self, act_id,held_mode):
    
        # yield LOOK_DOWN
        # yield LOOK_DOWN
        # reachable = self._get_reachable_area()
        # step_pix = int(args.STEP_SIZE / args.map_resolution)
        # pooled_reachable = pool2d(reachable, kernel_size=step_pix, stride=step_pix, pool_mode='avg')
        # state_xy = self.mapper.get_position_on_map()
        # shape_y = self.mapper.map_sz
        # state_x, state_y = state_xy
        # pooled_state_x, pooled_state_y = int(state_xy / step_pix)
        # if not held_mode:
        #     yield LOOK_DOWN
        #     yield LOOK_DOWN
        #     yield LOOK_UP
        #     yield LOOK_UP
        if act_id == MOVE_AHEAD:
            # if reachable[max(state_y - step_pix, 0): state_y, max(state_x - step_pix, 0): state_x].sum() > \
            #     reachable[max(state_y - step_pix, 0): state_y, state_x: min(state_x + step_pix, shape_x)].sum():
            yield MOVE_LEFT
            # else:
            #     yield MOVE_LEFT
        elif act_id == MOVE_LEFT:
            yield MOVE_BACK
        elif act_id == MOVE_BACK:
            yield MOVE_RIGHT
        else:
            yield MOVE_AHEAD

    def update_obstructed_actions(self, act_id: int) -> None:
        prev_len = len(self.obstructed_actions)
        if prev_len > 4000:
            pass
        else:
            for idx in range(prev_len):
                obstructed_acts = self.obstructed_actions[idx]
                self.obstructed_actions.append(obstructed_acts + [act_id])
            self.obstructed_actions.append([act_id])
    
    def update_global_point_cloud(self,curr_point_cloud):
        
        local_point_cloud = np.trunc(curr_point_cloud * 100) / 100
        local_point_cloud = local_point_cloud.reshape(-1,3)
        self.local_point_cloud = {(row[0], row[1], row[2]): row for row in local_point_cloud if row[2]<1.9}
        if len(self.global_point_cloud) == 0:
            self.global_point_cloud = {(row[0], row[1], row[2]): row for row in local_point_cloud if row[2]<1.9}
        else:
            for row in local_point_cloud:
                if row[2] > 1.9:
                    continue
                key = (row[0], row[1], row[2])
                if key not in self.global_point_cloud:
                    self.global_point_cloud[key] = row


    def explore_env(self):
        prev_act_id = 0
        num_sampled = 0
        if self.curr_stage == 'walkthrough':
            explore_policy = self.solution_config['walkthrough_search']
        else:
            explore_policy = self.solution_config['unshuffle_search']
        
        while step < self.step_max and (explore_policy == 'cover_continue' or prev_act_id != DONE or (explore_policy == "same" and step < len(self.walkthrough_action_list))):
            
            rgb_prev = self.rgb
            depth_prev = self.depth
            if args.use_seg:
                # seg_prev: H * W * (num_category + 1), segmented_dict:{'scores':, 'categories', 'masks'} 
                seg_prev, segmented_dict = self.segHelper.get_seg_pred(rgb_prev)
            else:
                seg_prev, segmented_dict = None, None

            # seg_prev_vis = visualize_segmentationRGB(rgb_prev, segmented_dict, visualize_sem_seg=True)
            # visdomImage([rgb_prev, seg_prev_vis], self.visdom, tag='subplot', info=['rgb', 'rgb_seg'],
                        #  max_depth=self.max_depth, prev_action=self.act_id_to_name[prev_act_id], current_step=self.step_from_stage_start)

            if self.solution_config["unshuffle_match"] != "feature_pointcloud_based":
                _,_,object_track_dict, curr_judge_new_and_centriod = self.mapper.add_observation(                                            
                                            self.position,
                                            self.rotation,
                                            -self.head_tilt,
                                            depth_prev,
                                            seg_prev,
                                            segmented_dict,
                                            self.object_tracker.get_objects_track_dict(),
                                            add_obs=True,
                                            add_seg=args.use_seg)

                self.object_tracker.set_objects_track_dict(object_track_dict)
                self.object_tracker.update_by_map_and_instance_and_feature(segmented_dict, curr_judge_new_and_centriod)
                # agent_pos = {"x":self.position["x"],
                #              "z":self.position["y"],
                #              "y":self.position["z"],
                #              "r":self.rotation,
                #              "t":self.head_tilt}
                # pos_dir = "./supplementary/agent_pos"
                # if not os.path.exists(pos_dir):
                #     os.mkdir(pos_dir)
                # with open(os.path.join(pos_dir,f"{step}_pos.json"),"w") as f:
                #     json.dump(agent_pos,f,indent=4)

                cur_isnan,cur_point_cloud,_,_ = self.mapper.add_observation(
                                            position=self.position,
                                            rotation=self.rotation,
                                            elevation=-self.head_tilt,
                                            depth=depth_prev,
                                            seg=seg_prev,
                                            seg_dict=segmented_dict,
                                            add_obs=True,
                                            add_seg=args.use_seg)
                if self.curr_stage == "walkthrough":
                    self.walkthrough_point_cloud_list.append(cur_point_cloud)
                    self.walkthrough_rgb_list.append(self.rgb)
                    self.walkthrough_point_cloud_isnan_list.append(cur_isnan)
                    self.walkthrough_depth_list.append(self.depth)
                    self.update_global_point_cloud(cur_point_cloud)
                    # # Image.fromarray(self.rgb).save(f"supplementary/walkthrough_rgb/{step}_walkthrough_rgb.jpg")
                    # # if step % 10 == 0:
                    # walkthrough_global_pointcloud = self.global_point_cloud.keys()
                    # point_cloud = np.array(list(walkthrough_global_pointcloud))
                    # vertices = [tuple(point) for point in point_cloud]
                    # vertex_element = PlyElement.describe(np.array(vertices, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]), 'vertex')
                    # PlyData([vertex_element]).write(f"supplementary/walkthrough_global_pointcloud/{step}_walkthrough_global_pointcloud.ply")
                    # walkthrough_local_pointcloud = self.local_point_cloud.keys()
                    # point_cloud = np.array(list(walkthrough_local_pointcloud))
                    # vertices = [tuple(point) for point in point_cloud]
                    # vertex_element = PlyElement.describe(np.array(vertices, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]), 'vertex')
                    # PlyData([vertex_element]).write(f"supplementary/walkthrough_local_pointcloud/{step}_walkthrough_local_pointcloud.ply")
                
                else:
                    walkthrough_point_cloud = self.walkthrough_point_cloud_list[step]
                    walkthrough_isnan = self.walkthrough_point_cloud_isnan_list[step]
                    walkthrough_rgb = self.walkthrough_rgb_list[step]
                    walkthrough_depth = self.walkthrough_depth_list[step]
                    self.unpdate_different_pointcloud(step,walkthrough_isnan,walkthrough_point_cloud,walkthrough_rgb,walkthrough_depth,cur_isnan,cur_point_cloud,self.rgb,self.depth)
                    self.update_global_point_cloud(cur_point_cloud)
                    # self.object_tracker.update(depth=depth_prev,segmented_dict=segmented_dict)
                    # Image.fromarray(self.rgb).save(f"supplementary/unshuffle_rgb/{step}_unshuffle_rgb.jpg")
                    # if step % 1 == 0:
                    # walkthrough_global_pointcloud = self.walkthrough_explorer.global_point_cloud.keys()
                    # walkthrough_add = set(self.walkthrough_explorer.global_point_cloud.keys())-set(self.global_point_cloud.keys())
                    # walkthrough_point_cloud = set(self.different_point_cloud["walkthrough"].keys())
                    # walkthrough_changed_point_cloud = list(walkthrough_add & walkthrough_point_cloud)
                    # point_cloud = np.array(walkthrough_changed_point_cloud)
                    # vertices = [tuple(point) for point in point_cloud]
                    # vertex_element = PlyElement.describe(np.array(vertices, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]), 'vertex')
                    # PlyData([vertex_element]).write(f"supplementary/walkthrough_changed_pointcloud/{step}_walkthrough_changed_pointcloud.ply")
                    # # point_cloud.save(f"{step}_walkthrough_changed_pointcloud.vtk")
                    # unshuffle_global_pointcloud = self.global_point_cloud.keys()
                    # point_cloud = np.array(list(unshuffle_global_pointcloud))
                    # vertices = [tuple(point) for point in point_cloud]
                    # vertex_element = PlyElement.describe(np.array(vertices, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]), 'vertex')
                    # PlyData([vertex_element]).write(f"supplementary/unshuffle_global_pointcloud/{step}_unshuffle_global_pointcloud.ply")
                    # # point_cloud.save(f"{step}_unshuffle_global_pointcloud.vtk")
                    # unshuffle_add = set(self.global_point_cloud.keys())-set(self.walkthrough_explorer.global_point_cloud.keys())
                    # unshuffle_point_cloud = set(self.different_point_cloud["unshuffle"].keys())
                    # unshuffle_changed_point_cloud = list(unshuffle_add & unshuffle_point_cloud)
                    # point_cloud = np.array(unshuffle_changed_point_cloud)
                    # vertices = [tuple(point) for point in point_cloud]
                    # vertex_element = PlyElement.describe(np.array(vertices, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]), 'vertex')
                    # PlyData([vertex_element]).write(f"supplementary/unshuffle_changed_pointcloud/{step}_unshuffle_changed_pointcloud.ply")
                    # # point_cloud.save(f"{step}_unshuffle_changed_pointcloud.vtk")
                    # unshuffle_local_pointcloud = self.local_point_cloud.keys()
                    # point_cloud = np.array(list(unshuffle_local_pointcloud))
                    # vertices = [tuple(point) for point in point_cloud]
                    # vertex_element = PlyElement.describe(np.array(vertices, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')]), 'vertex')
                    # PlyData([vertex_element]).write(f"supplementary/unshuffle_local_pointcloud/{step}_unshuffle_local_pointcloud.ply")
                


            
            
            # if self.curr_stage == "walkthrough":
            #     if step < len(self.walk_actions):
            #         act_id = self.act_name_to_id[self.walk_actions[step]]
            #     else:
            #         break
            if self.solution_config["unshuffle_search"] == "same" and self.curr_stage == "unshuffle":
                if step < len(self.walkthrough_action_list):
                    act_id = self.walkthrough_action_list[step]
                else:
                    break
                if self.acts == None:
                    act_id = None
                else:
                    act_id = next(self.acts, None)
                if act_id is None:
                    while self.execution.qsize() > 0:
                        op = self.execution.get()
                        self.acts = op()
                        if self.acts is not None:
                            act_id = next(self.acts, None)
                            if act_id is not None:
                                break
                if act_id is None:
                    act_id = DONE
            prev_act_id = act_id
            if self.solution_config["unshuffle_search"] == "same" and self.curr_stage == "walkthrough":
                self.walkthrough_action_list.append(act_id)
            print("Step ", self.step_from_stage_start, self.act_id_to_name[act_id])
            seg_prev_vis = visualize_segmentationRGB(rgb_prev, segmented_dict, visualize_sem_seg=True)
            
            # Image.fromarray(seg_prev_vis).save(f"supplementary/{self.curr_stage}_seg_img/{step}_seg_img.jpg")
            # Image.fromarray(rgb_prev).save(f"supplementary/{self.curr_stage}_seg_img/{step}_rgb.jpg")
            if self.vis != None:
                if args.use_seg:
                    seg_prev_vis = visualize_segmentationRGB(rgb_prev, segmented_dict, visualize_sem_seg=True)
                else:
                    seg_prev_vis = None

                if self.step_from_stage_start == 0:
                    self.vis.add_frame(image=rgb_prev, depth=depth_prev, seg_image=seg_prev_vis, add_map=True, add_map_seg=True, mapper=self.mapper,
                                       point_goal=self.point_goal, selem=self.selem, selem_agent_radius=self.selem_agent_radius, path=self.path, step=self.step_from_stage_start,
                                       stage=self.curr_stage, visdom=self.visdom, action=self.act_id_to_name[act_id])
                else:
                    self.vis.add_frame(image=rgb_prev, depth=depth_prev, seg_image=seg_prev_vis, add_map=True, add_map_seg=True, mapper=self.mapper,
                                       point_goal=self.point_goal, selem=self.selem, selem_agent_radius=self.selem_agent_radius, path=self.path, step=self.step_from_stage_start,
                                       stage=self.curr_stage, visdom=self.visdom, action=self.act_id_to_name[act_id])
            
            # plt.figure('View_distance')
            # step_pix = int(args.STEP_SIZE / args.map_resolution)
            # m_vis = np.invert(self.mapper.get_traversible_map(self.selem_agent_radius, 1,loc_on_map_traversible=True))
            # plt.imshow(m_vis * 0.8, origin='lower', vmin=0, vmax=1, cmap='Greys')
            # map_for_distance = self.mapper.get_map_for_view_distance()
            # norm = plt.Normalize(vmin = 0, vmax = 5)
            # plt.imshow(map_for_distance, alpha = 0.7, origin='lower', interpolation='nearest', cmap=plt.get_cmap('jet'), norm = norm)
            # # state_xy = self.mapper.get_position_on_map()
            # # state_theta = self.mapper.get_rotation_on_map()
            # # arrow_len = 1.0/self.mapper.resolution
            # # plt.arrow(state_xy[0], state_xy[1], 
            # #             arrow_len*np.cos(state_theta+np.pi/2),
            # #             arrow_len*np.sin(state_theta+np.pi/2), 
            # #             color='r', head_width=6)
            
            # if self.point_goal is not None:
            #     plt.plot(self.point_goal[1], self.point_goal[0], color='blue', marker='o',linewidth=10, markersize=4)
        
            # # plt.plot(state_xy[0], state_xy[1], color='r',markersize = 8)
            
        
            # ax=plt.gca()
            # path = os.path.join("supplementary/viewdist", f'{step}_viewDist.jpg')
            # plt.savefig(path)
            # print(f"saving {path}")
            # plt.close('View_distance')
            # if self.curr_stage == "unshuffle":
            #     top_image = drawTopImg(self.controller)
            #     Image.fromarray(top_image).save(f"supplementary/unshuffle_top_image/{step}_topImage.jpg")
            
            act_name = self.act_id_to_name[act_id]
            
            self.controller.step(action=act_name)
            
            self.rgb = self.controller.last_event.frame
            self.depth = self.controller.last_event.depth_frame
            # pdb.set_trace()

            act_isSuccess = self.check_successful_action(
                rgb=self.rgb, rgb_prev=rgb_prev, perc_diff_thresh=0.05) == True
            if act_isSuccess:
                self.update_position_and_rotation(act_id)
            else:
                if explore_policy != "same":
                    if self.init_on == False:
                        if 'Move' in self.act_id_to_name[act_id]:
                            self.mapper.add_obstacle_in_front_of_agent(
                                act_name=act_name, rotation=self.rotation)
                            # pass # lyy!
                        self.reset_execution_and_acts(act_id)
                if args.debug_print:
                    print("ACTION FAILED.", self.controller.last_event.metadata['errorMessage'])

            if explore_policy == 'cover_continue' and act_id == DONE:
                if args.debug_print:
                    print("num_sampled", num_sampled)
                num_sampled += 1
                if num_sampled > 5:
                    break
                ind_i, ind_j = self._sample_point_in_reachable_reachable()
                if not ind_i:
                    break
                # fn = lambda: self._cover_fn()
                # self.execution.put(fn)
                self.point_goal = ind_i, ind_j
                fn = lambda: self._point_goal_fn_assigned(goal_loc_cell=np.array(
                    [ind_j, ind_i]), dist_thresh=0.5, explore_mode=True)
                self.execution.put(fn)


            step += 1
            self._increment_num_steps_taken()
            self._update_metrics(action_name=act_name, action_success=act_isSuccess, calReward=False)

            # if step % 10 == 0:
            period_steps_num = 50
            if self.step_from_stage_start % period_steps_num == 0 and (args.test_mode == 'only_walkthrough' or args.test_mode == 'only_walkthrough_steps'):
                period_num = int(self.step_from_stage_start / period_steps_num)
                period_metrics = self.get_period_metrics_walkthrough()
                self.metrics[f'period_{period_num}'] = period_metrics
            
    def unpdate_different_pointcloud(self,step,walkthrough_isnan,walkthrough_point_cloud,walkthrough_rgb,walkthrough_depth,cur_isnan,cur_point_cloud,cur_rgb,cur_depth):
        
        diff = np.abs(walkthrough_depth - cur_depth)
        diff_mask = np.where(diff > 0.005, 1, 0).astype(bool)
        # rgb_diff = cv2.absdiff(walkthrough_rgb, cur_rgb)
    
        # gray_difference = cv2.cvtColor(rgb_diff, cv2.COLOR_BGR2GRAY)
    
        # _, rgb_diff_mask = cv2.threshold(gray_difference, 20, 255, cv2.THRESH_BINARY)
        # rgb_diff_mask = rgb_diff_mask.astype(bool)

        # diff_mask = diff_mask & rgb_diff_mask

        mask_walk = walkthrough_depth < cur_depth
        mask_unshuffle = walkthrough_depth > cur_depth
        mask_walk = diff_mask & mask_walk
        mask_unshuffle = diff_mask & mask_unshuffle
        mask_walk = mask_walk & np.logical_not(walkthrough_isnan) & np.logical_not(cur_isnan)
        mask_walk = mask_walk | (np.logical_not(walkthrough_isnan) & cur_isnan)
        mask_unshuffle = mask_unshuffle & np.logical_not(walkthrough_isnan) & np.logical_not(cur_isnan)
        mask_unshuffle = mask_unshuffle | (np.logical_not(cur_isnan) & walkthrough_isnan)
        walk_objs_bounding_boxes,walk_objs_masks,mask_walk_opened = self.find_bounding_boxes_and_masks(mask_walk)
        unshuffle_objs_bounding_boxes,unshuffle_objs_masks,mask_unshuffle_opened = self.find_bounding_boxes_and_masks(mask_unshuffle)
        walk_objs_features = self.extract_features_of_every_box(walkthrough_rgb,walk_objs_bounding_boxes)
        unshuffle_objs_features = self.extract_features_of_every_box(cur_rgb,unshuffle_objs_bounding_boxes)
        
        if len(walk_objs_bounding_boxes) or len(unshuffle_objs_bounding_boxes):
            rgb = np.hstack((walkthrough_rgb, cur_rgb))
            mask = np.hstack((mask_walk,mask_unshuffle))
            morphologyEx_mask = np.hstack((mask_walk_opened.astype(bool),mask_unshuffle_opened.astype(bool)))
            image = Image.fromarray(rgb)
            image.save(f"/home/lyy/rearrange_on_ProcTHOR/test/{step}.jpg")
            image_mask = Image.fromarray(mask)
            image_mask.save(f"/home/lyy/rearrange_on_ProcTHOR/test/mask_{step}.jpg")
            image_morphologyEx_mask = Image.fromarray(morphologyEx_mask)
            image_morphologyEx_mask.save(f"/home/lyy/rearrange_on_ProcTHOR/test/morphologyEx_mask_{step}.jpg")
        if len(walk_objs_features):
            walkthrough_point_cloud = np.trunc(walkthrough_point_cloud * 100) / 100
            walk_diff_pointcloud = np.zeros((walkthrough_point_cloud.shape[0],walkthrough_point_cloud.shape[1],4))
            for i in range(len(walk_objs_masks)):
                obj_mask = walk_objs_masks[i]
                walk_diff_pointcloud[obj_mask,:3] = walkthrough_point_cloud[obj_mask,:]
                feature_id = len(self.feature_list)
                walk_diff_pointcloud[obj_mask,3] = feature_id
                feature = walk_objs_features[i]
                self.feature_list.append(feature)
            walk_diff_pointcloud = walk_diff_pointcloud.reshape(-1,4)
            if len(self.different_point_cloud["walkthrough"]) == 0:
                self.different_point_cloud["walkthrough"] = {(row[0], row[1], row[2]): [row[3]] for row in walk_diff_pointcloud}
            else:
                for row in walk_diff_pointcloud:
                    key = (row[0], row[1], row[2])
                    if key in self.different_point_cloud["walkthrough"]:
                        self.different_point_cloud["walkthrough"][key].append(row[3])
                    else:
                        self.different_point_cloud["walkthrough"][key] = [row[3]]
        if len(unshuffle_objs_features):
            unshuffle_point_cloud = np.trunc(cur_point_cloud * 100) / 100
            
            # print(np.max(unshuffle_point_cloud))
            unshuffle_diff_pointcloud = np.zeros((unshuffle_point_cloud.shape[0],walkthrough_point_cloud.shape[1],4))
            for i in range(len(unshuffle_objs_masks)):
                obj_mask = unshuffle_objs_masks[i]
                unshuffle_diff_pointcloud[obj_mask,:3] = unshuffle_point_cloud[obj_mask,:]
                feature_id = len(self.feature_list)
                unshuffle_diff_pointcloud[obj_mask,3] = feature_id
                feature = unshuffle_objs_features[i]
                self.feature_list.append(feature)
            unshuffle_diff_pointcloud = unshuffle_diff_pointcloud.reshape(-1,4)
            if len(self.different_point_cloud["unshuffle"]) == 0:
                self.different_point_cloud["unshuffle"] = {(row[0], row[1], row[2]): [row[3]] for row in unshuffle_diff_pointcloud}
            else:
                for row in unshuffle_diff_pointcloud:
                    key = (row[0], row[1], row[2])
                    if key in self.different_point_cloud["unshuffle"]:
                        self.different_point_cloud["unshuffle"][key].append(row[3])
                    else:
                        self.different_point_cloud["unshuffle"][key] = [row[3]]

        
    def extract_features_of_every_box(self,rgb,bounding_boxes):
        objs_feature = []
        img = Image.fromarray(rgb)
        height, width = img.size
        img_tensor = self.transform(img).unsqueeze(0)
        img_feature = self.feature_extractor(img_tensor)
        for bbox in bounding_boxes:
            x_min,y_min,x_max,y_max = bbox
            bbox = torch.tensor([bbox])
            normalized_bbox = torch.zeros_like(bbox, dtype=img_feature.dtype)
            normalized_bbox[0, 0] = bbox[0, 0] / width
            normalized_bbox[0, 1] = bbox[0, 1] / height
            normalized_bbox[0, 2] = bbox[0, 2] / width
            normalized_bbox[0, 3] = bbox[0, 3] / height
            ins_visual_feature = ops.roi_pool(img_feature, [normalized_bbox], output_size=(1,1)).ravel()
            adjust_bbox = (x_min, y_min, x_max, y_max)
            ins_color_feature = get_color_feature(rgb, adjust_bbox)
            concat_feature = torch.cat((ins_visual_feature, ins_color_feature)).ravel().cpu().detach().numpy()
            objs_feature.append(concat_feature)
        return objs_feature

    def find_bounding_boxes_and_masks(self,mask_image):
        '''gei ding yi'''
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        opened = cv2.morphologyEx(mask_image.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes = []
        masks = []
        width,height = mask_image.shape
        if len(contours):
            for contour in contours:
                mask = np.zeros_like(mask_image).astype(np.uint8)
                cv2.fillPoly(mask, [contour], 255)
                masks.append(mask.astype(bool))
                x, y, w, h = cv2.boundingRect(contour)
                bounding_boxes.append([x, y, x+w, y+h])
        return bounding_boxes,masks,opened
            
    def move_all_maps_to_center(self):
        pass

    def _sample_point_in_reachable_reachable(self):
        reachable = self._get_reachable_area()
        state_xy = self.mapper.get_position_on_map()

        inds_i, inds_j = np.where(reachable)
        dist = np.sqrt(np.sum(np.square(np.expand_dims(
            state_xy, axis=1) - np.stack([inds_j, inds_i], axis=0)), axis=0))
        sorted_dist_indices = np.argsort(dist)
        dist = dist[sorted_dist_indices]
        sorted_inds_i = inds_i[sorted_dist_indices]
        sorted_inds_j = inds_j[sorted_dist_indices]
        
        # dist_thresh = dist > 20.0
        # inds_i = inds_i[dist_thresh]
        # inds_j = inds_j[dist_thresh]
        if inds_i.shape[0] == 0:
            print("FOUND NO REACHABLE INDICES")
            return [], []
        
        rng = np.random.RandomState(self.seed)
        self.seed+=1
        ind = -rng.randint(10)
        # ind_i, ind_j = inds_i[ind], inds_j[ind]
        ind_i, ind_j = sorted_inds_i[ind], sorted_inds_j[ind]
        return ind_i, ind_j

    def get_curr_pose_difference_energy(self, stage):
        curr_objs_id_to_pose = self.get_current_objs_id_to_pose(stage=stage)
        curr_energies_dict = pose_difference_energy(
            goal_poses=self.walkthrough_objs_id_to_pose, cur_poses=curr_objs_id_to_pose)
        return curr_energies_dict

    def get_metrics(self):
        if self.curr_stage == 'walkthrough':
            n_reachable = len(self.reachable_positions_in_simulator)
            n_obj_seen = len(self.seen_openable_objects) + \
                len(self.seen_pickupable_objects)
            self.metrics = {
                **self.metrics,
                **{
                    f'{self.curr_stage}/ep_length': self.step_from_stage_start,
                    f'{self.curr_stage}/num_explored_xz': len(self.visited_xz),
                    f'{self.curr_stage}/num_explored_xzr': len(self.visited_xzr),
                    f'{self.curr_stage}/prop_visited_xz': len(self.visited_xz) / n_reachable,
                    f'{self.curr_stage}/prop_visited_xzr': len(self.visited_xzr) / (int(360 / args.DT) * n_reachable),
                    f'{self.curr_stage}/num_obj_seen': n_obj_seen,
                    f'{self.curr_stage}/prop_obj_seen': n_obj_seen / self.total_pickupable_or_openable_objects,
                }
            }

        return self.metrics

    def get_period_metrics_walkthrough(self):
        period_metrics = {}
        if self.curr_stage == 'walkthrough':
            n_reachable = len(self.reachable_positions_in_simulator)
            n_obj_seen = len(self.seen_openable_objects) + len(self.seen_pickupable_objects)
            
            period_metrics = {
                    f'{self.curr_stage}/ep_length': self.step_from_stage_start,
                    f'{self.curr_stage}/num_explored_xz': len(self.visited_xz),
                    f'{self.curr_stage}/num_explored_xzr': len(self.visited_xzr),
                    f'{self.curr_stage}/prop_visited_xz': len(self.visited_xz) / n_reachable,
                    f'{self.curr_stage}/prop_visited_xzr': len(self.visited_xzr) / (int(360 / args.DT) * n_reachable),
                    f'{self.curr_stage}/num_obj_seen': n_obj_seen,
                    f'{self.curr_stage}/prop_obj_seen': n_obj_seen / self.total_pickupable_or_openable_objects,
                }
        return period_metrics


