from explore import Explore
from ai2thor.controller import Controller
from visualization import Animation
import numpy as np
from arguments import args
from planner import FMMPlanner
from queue import LifoQueue
from visualization import visualize_segmentationRGB
from rearrange_on_proc.constants import STOP, ROTATE_LEFT, ROTATE_RIGHT, MOVE_AHEAD, MOVE_BACK, MOVE_LEFT, MOVE_RIGHT, DONE, LOOK_DOWN, PICKUP, OPEN, CLOSE, PUT, DROP, LOOK_UP
from rearrange_on_proc.constants import CATEGORY_LIST, CATEGORY_to_ID, INSTANCE_FEATURE_SIMILARITY_THRESHOLD, INSTANCE_IOU_THRESHOLD, INSTANCE_CENTROID_THRESHOLD
from typing import Dict, Any, Tuple, Optional, Callable, List, Union, Sequence
from rearrange_on_proc.utils.utils import visdomImage, ObjectInteractablePostionsCache, pool2d, are_poses_equal,floyd, cosine_similarity, calculate_iou_numpy
from rearrange_on_proc.utils.geom import eul2rotm_py
from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment
from scipy.spatial import distance
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from object_tracker import ObjectTrack
from numpy import ma
import scipy
import skfmm
import networkx as nx
from relation import Relations_CenterOnly, Relation_Calculator
import utils
from collections import deque
from rearrange_on_proc.point_cloud import point_cloud_instance_seg,point_cloud_seg_visualization
from sklearn.cluster import OPTICS,DBSCAN
import rearrange_on_proc.utils.utils as utils
from scipy.spatial import cKDTree
from collections import Counter
from pointnet.pointnet2_helper import PointNet2Helper
from allenact_plugins.ithor_plugin.ithor_util import include_object_data


class Rearrange():
    def __init__(self, 
                 task_id,
                 walkthrouth_explorer: Explore = None, 
                 unshuffle_explorer: Explore = None, 
                 process_id=None, 
                 task: dict = None, 
                 visdom=None, 
                 walkthrough_objs_id_to_pose=None, 
                 unshuffle_objs_id_to_pose=None, 
                 solution_config=None,
                 gpu_id=0,
                 ) -> None:
        self.task_id = task_id
        self.gpu_id = gpu_id
        self.walkthrough_explorer = walkthrouth_explorer
        self.unshuffle_explorer = unshuffle_explorer
        self.object_to_rearrange_dict_list = []
        self.solution_config = solution_config
        # print(f"---:p{process_id}:{task['unique_id']} ---「rearrange stage」---")

        self.task = task
        self.visdom = visdom
        # Cache of where objects can be interacted with
        self._interactable_positions_cache = ObjectInteractablePostionsCache()
        self.walkthrough_objs_id_to_pose = walkthrough_objs_id_to_pose
        self.unshuffle_objs_id_to_pose = unshuffle_objs_id_to_pose
        self.error_message = None
        self.rearrange_pickupable = [
                obj["objectType"] for obj in self.unshuffle_explorer.controller.last_event.metadata["objects"] if obj["pickupable"]]
        self.rearrange_pickupable = [
                obj for obj in self.rearrange_pickupable if obj in CATEGORY_LIST]

            
        self.receptacles = [obj["objectType"] for obj in self.unshuffle_explorer.controller.last_event.metadata["objects"] if obj["receptacle"]]
        self.receptacles = [obj for obj in self.receptacles if obj in CATEGORY_LIST]
        # self.relation_calculator = Relation_Calculator(self.rearrange_pickupable, self.receptacles)

        self.open_objects_list = []

        self.pointnet2_helper = PointNet2Helper(device_id=self.gpu_id)

    def get_metrics(self):
        return {
            **self.unshuffle_explorer.metrics,
            'unshuffle/ep_length': self.unshuffle_explorer.step_from_stage_start,
        }

    def match_two_map_from_category(self, walkthrough_map: np.ndarray, unshuffle_map: np.ndarray, dist_thresh):
        """
        output:dict{category:(current_pos,origin_pos)}
        """
        difference = unshuffle_map.astype(bool).astype(float) - walkthrough_map.astype(bool).astype(float)

        for category_id in range(1, difference.shape[3]):
            if CATEGORY_LIST[category_id-1] not in self.rearrange_pickupable:
                continue
            cur_indices = np.where(difference[:, :, :, category_id] > 0)
            cur_ind_i, cur_ind_j, cur_ind_k = cur_indices

            origin_indices = np.where(difference[:, :, :, category_id] < 0)
            origin_ind_i, origin_ind_j, origin_ind_k = origin_indices

            move_distance = np.linalg.norm(np.array(
                [cur_i_center, cur_j_center, cur_k_center]) - np.array([origin_i_center, origin_j_center, origin_k_center]))
            if move_distance > dist_thresh:
                obj_info = {}
                obj_info["label"] = category_id
                obj_info['cur_map_pos'] = np.array([cur_i_center, cur_j_center, cur_k_center])
                obj_info['origin_map_pos'] = np.array([origin_i_center, origin_j_center, origin_k_center])
                obj_info['cur_ins_id'] = unshuffle_map[int(cur_i_center), int(cur_j_center), int(cur_k_center), category_id]
                obj_info['origin_ins_id'] = walkthrough_map[int(origin_i_center), int(origin_j_center), int(origin_k_center), category_id]

                self.object_to_rearrange_dict_list.append(obj_info)

    def greedy_compute_distance_matrix(self) -> np.ndarray:
        '''
        '''
        distance_matrix = np.zeros(
            (len(self.object_to_rearrange_dict_list) + 1, len(self.object_to_rearrange_dict_list) + 1))
        start_position_map = self.unshuffle_explorer.mapper.get_position_on_map()
        for i in range(len(self.object_to_rearrange_dict_list)):
            cur_map_pos_2d = np.array([self.object_to_rearrange_dict_list[i]['cur_map_pos']
                                      [1], self.object_to_rearrange_dict_list[i]['cur_map_pos'][0]])
            origin_map_pos_2d = np.array([self.object_to_rearrange_dict_list[i]['origin_map_pos']
                                         [1], self.object_to_rearrange_dict_list[i]['origin_map_pos'][0]])
            dis_from_start_to_object_i = np.linalg.norm(start_position_map - cur_map_pos_2d) + np.linalg.norm(
                cur_map_pos_2d - origin_map_pos_2d)
            distance_matrix[0][i + 1] = dis_from_start_to_object_i
        for i in range(len(self.object_to_rearrange_dict_list)):
            for j in range(len(self.object_to_rearrange_dict_list)):
                if i == j:
                    continue
                origin_map_pos_i_2d = np.array(
                    [self.object_to_rearrange_dict_list[i]['origin_map_pos'][1], self.object_to_rearrange_dict_list[i]['origin_map_pos'][0]])
                cur_map_pos_j_2d = np.array(
                    [self.object_to_rearrange_dict_list[j]['cur_map_pos'][1], self.object_to_rearrange_dict_list[j]['cur_map_pos'][0]])
                origin_map_pos_j_2d = np.array(
                    [self.object_to_rearrange_dict_list[j]['origin_map_pos'][1], self.object_to_rearrange_dict_list[j]['origin_map_pos'][0]])
                dis_from_i_to_j = np.linalg.norm(origin_map_pos_i_2d - cur_map_pos_j_2d) + np.linalg.norm(
                    cur_map_pos_j_2d - origin_map_pos_j_2d)
                distance_matrix[i + 1][j + 1] = dis_from_i_to_j
        return distance_matrix

    def greedy_reorder(self) -> list:
        '''
        '''
        distance_matrix = self.greedy_compute_distance_matrix()
        nodes = list(range(distance_matrix.shape[0]))
        start_node = 0
        nodes.remove(start_node)
        path = []
        current_node = start_node

        while nodes:
            next_city = min(
                nodes, key=lambda city: distance_matrix[current_node][city])
            nodes.remove(next_city)
            path.append(next_city - 1)
            current_node = next_city

        return path
    
    def random_reorder(self):
        np.random.seed(0)
        np.random.shuffle(arr)
        return list(arr)

    def or_tools_reorder(self) -> list:
        distance_matrix = self.or_tools_compute_distance_matrix()
        if args.debug_print:
            print("distance_matrix", distance_matrix)
        constraints = self.compute_topo_constraints()
        if args.debug_print:
            print("constraints", constraints)
        data = self.or_tools_create_data_model(
            distance_matrix=distance_matrix, constraints=constraints)
        route = self.or_tools_compute_rearrage_order(data)
        if args.debug_print:
            print(route)
        order = []
        for i in range(1, len(route)):
            if route[i] % 2 == 0:
                # order.append(int(route[i] / 2 - 1))
                order.append(self.pointsId_to_rearrangeDictId[route[i]])
        return order

    def or_tools_compute_rearrage_order(self, data):
        """Entry point of the program."""

        # Create the routing index manager.
        manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                               data['num_vehicles'], data['depot'])

        # Create Routing Model.
        routing = pywrapcp.RoutingModel(manager)

        # Define cost of each arc.

        def distance_callback(from_index, to_index):
            """Returns the manhattan distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['distance_matrix'][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(
            distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Add Distance constraint.
        dimension_name = 'Distance'
        routing.AddDimension(
            transit_callback_index,
            0,  # no slack
            np.iinfo(np.int64).max,  # vehicle maximum travel distance
            True,  # start cumul to zero
            dimension_name)
        distance_dimension = routing.GetDimensionOrDie(dimension_name)

        # Add Capacity constraint.
        def demand_callback(from_index):
            """Returns the demand of the node."""
            # Convert from routing variable Index to demands NodeIndex.
            from_node = manager.IndexToNode(from_index)
            return data['demands'][from_node]

        demand_callback_index = routing.RegisterUnaryTransitCallback(
            demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            data['vehicle_capacities'],  # vehicle maximum capacities
            True,  # start cumul to zero
            'Capacity')

        # Define Transportation Requests.
        for request in data['pickups_deliveries']:
            pickup_index = manager.NodeToIndex(request[0])
            delivery_index = manager.NodeToIndex(request[1])
            routing.AddPickupAndDelivery(pickup_index, delivery_index)
            routing.solver().Add(
                routing.VehicleVar(pickup_index) == routing.VehicleVar(
                    delivery_index))
            routing.solver().Add(
                distance_dimension.CumulVar(pickup_index) <=
                distance_dimension.CumulVar(delivery_index))
        # Define Rearrange Topo Requests
        for request in data['topo_constraints']:
            pickup_index = manager.NodeToIndex(request[0])
            delivery_index = manager.NodeToIndex(request[1])
            routing.solver().Add(
                routing.VehicleVar(pickup_index) == routing.VehicleVar(
                    delivery_index))
            routing.solver().Add(
                distance_dimension.CumulVar(pickup_index) <=
                distance_dimension.CumulVar(delivery_index))
        if args.debug_print:
            print('OR-tools debug1')
        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        # search_parameters.log_search = True
        # search_parameters.solution_limit = 100
        search_parameters.time_limit.FromSeconds(10)
        if args.debug_print:
            print('OR-tools debug2')
        # Solve the problem.
        solution = routing.SolveWithParameters(search_parameters)
        if args.debug_print:
            print('OR-tools debug3')

        # Get vehicle routes from a solution and store them in an array
        if solution:
            route = self.get_routes(solution, routing, manager)[0][:-1]
        else:
            route = []
        return route

    def get_routes(self, solution, routing, manager):
        """Get vehicle routes from a solution and store them in an array."""
        # Get vehicle routes and store them in a two dimensional array whose
        # i,j entry is the jth location visited by vehicle i along its route.
        routes = []
        for route_nbr in range(routing.vehicles()):
            index = routing.Start(route_nbr)
            route = [manager.IndexToNode(index)]
            while not routing.IsEnd(index):
                index = solution.Value(routing.NextVar(index))
                route.append(manager.IndexToNode(index))
            routes.append(route)
        return routes

    def or_tools_create_data_model(self, distance_matrix, constraints):
        """Stores the data for the problem."""
        data = {}
        data['distance_matrix'] = distance_matrix
        data['demands'] = [0]
        data['pickups_deliveries'] = []
        for i in range(len(self.object_to_rearrange_dict_list)):
            pickup_delivery = [2 * i + 1, 2 * i + 2]
            data['pickups_deliveries'].append(pickup_delivery)
            data['demands'].append(1)
            data['demands'].append(-1)

        data['topo_constraints'] = []
        for obj_i in range(len(constraints)):
            for obj_j in constraints[obj_i]:
                topo_constraint = [2 * obj_i + 1, 2 * obj_j + 1]
                data['topo_constraints'].append(topo_constraint)
        data['num_vehicles'] = 1
        data['depot'] = 0
        data['vehicle_capacities'] = [1]
        return data

    def compute_topo_constraints(self, dist_thresh=0.4):
        """
        constraints = [[]
                       for _ in range(len(self.object_to_rearrange_dict_list))]
        for i in range(0, len(self.object_to_rearrange_dict_list) - 1):
            for j in range(i + 1, len(self.object_to_rearrange_dict_list)):
                obj_i_cur_pos = self.object_to_rearrange_dict_list[i]["cur_map_pos"]
                obj_i_origin_pos = self.object_to_rearrange_dict_list[i]["origin_map_pos"]
                obj_j_cur_pos = self.object_to_rearrange_dict_list[j]["cur_map_pos"]
                obj_j_origin_pos = self.object_to_rearrange_dict_list[j]["origin_map_pos"]
                if np.linalg.norm(obj_i_cur_pos - obj_j_origin_pos) * args.map_resolution < dist_thresh:
                    constraints[i].append(j)
                elif np.linalg.norm(obj_j_cur_pos - obj_i_origin_pos) * args.map_resolution < dist_thresh:
                    constraints[j].append(i)
        return constraints

    def or_tools_compute_distance_matrix(self) -> List[List[int]]:
        points = []
        agent_pos_on_map = self.unshuffle_explorer.mapper.get_position_on_map().astype(np.int32)
        agent_pos_on_map = (agent_pos_on_map[1], agent_pos_on_map[0])
        points.append(agent_pos_on_map)

        self.pointsId_to_rearrangeDictId = {}
        pointsId = 2
        for rearrangeDictId, obj_dict in enumerate(self.object_to_rearrange_dict_list):
            obj_cur_pos_on_map = (
                obj_dict["cur_map_pos"][0], obj_dict["cur_map_pos"][1])
            # point = self.unshuffle_explorer.get_clostest_reachable_map_pos(obj_cur_pos_on_map)
            # point = self.unshuffle_explorer.choose_reachable_map_pos_in_same_room(
            #     obj_cur_pos_on_map, 1.5 + args.STEP_SIZE,within_thresh=False)
            point = self.unshuffle_explorer.choose_reachable_map_pos_in_same_room(
                obj_cur_pos_on_map, 1.0,within_thresh=False)
            if point is None:
                continue
            points.append(point)
            obj_origin_pos_on_map = (
                obj_dict["origin_map_pos"][0], obj_dict["origin_map_pos"][1])
            # point = self.unshuffle_explorer.get_clostest_reachable_map_pos(obj_origin_pos_on_map)
            # point = self.unshuffle_explorer.choose_reachable_map_pos_in_same_room(
            #     obj_origin_pos_on_map, 1.5 + args.STEP_SIZE,within_thresh=False)
            point = self.unshuffle_explorer.choose_reachable_map_pos_in_same_room(
                obj_origin_pos_on_map, 1.0,within_thresh=False)
            if point is None:
                points.pop()
                continue
            points.append(point)
            self.pointsId_to_rearrangeDictId[pointsId] =  rearrangeDictId
            pointsId += 2

        if args.debug_print:
            print("points", points)
        grid = self.unshuffle_explorer._get_reachable_area()
        distance_matrix = floyd(grid=grid,points=points)
        # # distance_matrix = distance_matrix.tolist()\
        # import compress_pickle
        # import os
        # compress_pickle.dump(
        #             obj=distance_matrix,
        #             path="distance_matrix.pkl.gz",
        #             pickler_kwargs={"protocol": 4,},  
        #         )

        return distance_matrix

    def _close_enough_fn_assigned(self, obj_position: np.ndarray, held_mode: bool, isFirstTime = False):
        if args.debug_print:
            print("_close_enough_fn_assigned")
        agent_position = self.unshuffle_explorer.mapper.get_position_on_map()
        dist = np.sqrt(np.sum(np.square(agent_position - obj_position)))
        if isFirstTime:
            threshold = 1.0
        else:
            threshold = 1.5
        close_enough = dist * self.unshuffle_explorer.mapper.resolution < threshold
        if close_enough:
            if args.debug_print:
                print(f"close enough to {obj_position} in {agent_position}")
        else:
            # point_goal = self.unshuffle_explorer.choose_reachable_map_pos_in_same_room(
            #     [obj_position[1], obj_position[0]], args.STEP_SIZE + 1.5,within_thresh=True)
            point_goal = self.unshuffle_explorer.choose_reachable_map_pos_in_same_room(
                [obj_position[1], obj_position[0]], 1.0,within_thresh=True)
            if point_goal is None:
                if args.debug_print:
                    print('None reachable position')
                return 
            ind_i, ind_j = point_goal
            reached = self.unshuffle_explorer._check_point_goal_reached(np.array([ind_j, ind_i]), dist_thresh=0.5)
            if reached:
                if args.debug_print:
                    print("reached",point_goal)
                return
            fn = lambda: self._close_enough_fn_assigned(
                obj_position, held_mode=held_mode, isFirstTime=False)
            self.unshuffle_explorer.execution.put(fn)

            self.unshuffle_explorer.point_goal = point_goal
            fn = lambda: self.unshuffle_explorer._point_goal_fn_assigned(np.array(
                [ind_j, ind_i]), explore_mode=False, dist_thresh=0.5, held_mode=held_mode)
            self.unshuffle_explorer.execution.put(fn)

    def get_fmm_clostest_reachable_map_pos(self, map_pos):
        reachable = self.unshuffle_explorer._get_reachable_area()
        step_pix = int(args.STEP_SIZE / args.map_resolution)
        pooled_map_pos = [map_pos[0] // step_pix, map_pos[1] // step_pix]
        pooled_reachable = pool2d(
            reachable, kernel_size=step_pix, stride=step_pix, pool_mode='avg')
        pooled_reachable = pooled_reachable == 1
        if args.debug_print:
            print("pooled_reachable_sum", pooled_reachable.sum())
        # visdomImage(pooled_reachable, self.visdom, tag='01')
            print("pooled_map_pos", pooled_map_pos)

        pooled_reachable_ma = ma.masked_values(pooled_reachable * 1, 0)
        if args.debug_print:
            print("pooled_reachable_ma[0][0]", pooled_reachable_ma[0][0])
        goal_x, goal_y = int(pooled_map_pos[1]), int(pooled_map_pos[0])
        goal_x = min(goal_x, pooled_reachable_ma.shape[1] - 1)
        goal_y = min(goal_y, pooled_reachable_ma.shape[0] - 1)
        goal_x = max(goal_x, 0)
        goal_y = max(goal_y, 0)
        pooled_reachable_ma[goal_y, goal_x] = 0
        dd = skfmm.distance(pooled_reachable_ma, dx=1)
        # print("dd[0][0]", dd[0][0])
        dd = ma.filled(dd, np.nan)
        dd_mask = np.invert(np.isnan(ma.filled(dd, np.nan)))
        # print("dd_sum", dd_mask.sum())
        sorted_indexes = np.argsort(dd.flatten())[:2]

        row_indexes, col_indexes = np.unravel_index(sorted_indexes, dd.shape)
        # print("row_indexes, col_indexes", row_indexes, col_indexes)
        min_row, min_col = row_indexes[1], col_indexes[1]
        # print("pooled_reachable[ind_i][ind_j]", spooled_reachable[min_row][min_col])
        ind_i, ind_j = min_row * step_pix, min_col * step_pix
        # print(ind_i, ind_j)
        # print("reachable[ind_i][ind_j]", reachable[ind_i][ind_j])
        return ind_i, ind_j

    def set_excution(self, obj_pos, held_mode):
        '''
        '''
        self.unshuffle_explorer.execution = LifoQueue(maxsize=200)
        self.unshuffle_explorer.acts = None
        map_pos = [obj_pos[0], obj_pos[1]]
        fn = lambda: self._close_enough_fn_assigned(
            obj_position=np.array([obj_pos[1], obj_pos[0]]), held_mode=held_mode, isFirstTime=True)
        self.unshuffle_explorer.execution.put(fn)

    def rearrange_reset_execution_and_acts(self, obj_pos, held_mode,act_id):
        self.unshuffle_explorer.execution = LifoQueue(maxsize=200)
        self.unshuffle_explorer.acts = None
        agent_position = self.unshuffle_explorer.mapper.get_position_on_map()
        goal_position = np.array([obj_pos[1], obj_pos[0]])
        dist = np.sqrt(np.sum(np.square(agent_position - goal_position)))
        if dist * self.unshuffle_explorer.mapper.resolution < 1.5:
            if args.debug_print:
                print("Close enough to goal position.Need not to reset execution")
            return
        # new_point_goal = self.unshuffle_explorer.choose_reachable_map_pos_in_same_room(
        #     [obj_pos[0], obj_pos[1]], thresh=1.5 + args.STEP_SIZE,within_thresh=True)
        new_point_goal = self.unshuffle_explorer.choose_reachable_map_pos_in_same_room(
            [obj_pos[0], obj_pos[1]], thresh=1.0,within_thresh=True)
        if new_point_goal == None:
            return

        fn = lambda: self._close_enough_fn_assigned(
            np.array([obj_pos[1], obj_pos[0]]), held_mode=held_mode, isFirstTime=True)
        self.unshuffle_explorer.execution.put(fn)

        self.unshuffle_explorer.point_goal = new_point_goal
        ind_i, ind_j = self.unshuffle_explorer.point_goal
        fn = lambda: self.unshuffle_explorer._point_goal_fn_assigned(
            goal_loc_cell=np.array([ind_j, ind_i]), dist_thresh=0.5, explore_mode=False, held_mode=held_mode)
        self.unshuffle_explorer.execution.put(fn)
        fn = lambda:self.unshuffle_explorer._random_move_fn(act_id=act_id,held_mode=held_mode)
        self.unshuffle_explorer.execution.put(fn)

    def reset_headtilt(self):
        act_seq = deque()
        headtilt = self.unshuffle_explorer.head_tilt
        if headtilt == 0:
            return
        elif headtilt == -30:
            act_seq.append(LOOK_DOWN)
        elif headtilt == 30:
            act_seq.append(LOOK_UP)
        elif headtilt == 60:
            act_seq.append(LOOK_UP)
            act_seq.append(LOOK_UP)

        while bool(act_seq):
            act_id = act_seq.pop()
            self.act(act_id)



    def navigate_to_object_pos(self, obj_pos, step_max, held_mode):
        self.set_excution(obj_pos=obj_pos, held_mode=held_mode)
        prev_act_id = 0


        while step < step_max and prev_act_id != DONE:
            act_seq = self.reset_headtilt()
                
            rgb_prev = self.unshuffle_explorer.rgb
            depth_prev = self.unshuffle_explorer.depth
            if args.use_seg:
                seg_prev, segmented_dict = self.unshuffle_explorer.segHelper.get_seg_pred(
                    rgb_prev)
            else:
                seg_prev, segmented_dict = None, None
            # object_track_dict, curr_judge_new = self.unshuffle_explorer.mapper.add_observation(self.unshuffle_explorer.position,
            #                                                self.unshuffle_explorer.rotation,
            #                                                -self.unshuffle_explorer.head_tilt,
            #                                                depth_prev,
            #                                                seg_prev,
            #                                                segmented_dict,
            #                                                self.unshuffle_explorer.object_tracker.get_objects_track_dict(),
            #                                                add_obs=True,
            #                                                add_seg=args.use_seg)
            # self.unshuffle_explorer.object_tracker.set_objects_track_dict(object_track_dict)

            #     self.unshuffle_explorer.object_tracker.update_by_category_and_distance(mapper = self.mapper , depth=depth_prev, segmented_dict=segmented_dict)
            # elif self.solution_config['unshuffle_match'] == 'instance_based' or self.solution_config['unshuffle_match'] == 'map_based':
            # self.unshuffle_explorer.object_tracker.update_by_map_and_instance_and_feature(segmented_dict, curr_judge_new)

            if self.unshuffle_explorer.acts == None:
                act_id = None
            else:
                act_id = next(self.unshuffle_explorer.acts, None)
            if act_id is None:
                while self.unshuffle_explorer.execution.qsize() > 0:
                    op = self.unshuffle_explorer.execution.get()
                    self.unshuffle_explorer.acts = op()
                    if self.unshuffle_explorer.acts is not None:
                        act_id = next(self.unshuffle_explorer.acts, None)
                        if act_id is not None:
                            break
            if act_id is None:
                act_id = DONE
            prev_act_id = act_id
            act_isSuccess = self.act(act_id = act_id, path = self.unshuffle_explorer.path, point_goal = self.unshuffle_explorer.point_goal)
            act_name = self.unshuffle_explorer.act_id_to_name[act_id]
            if not act_isSuccess:
                if 'Move' in act_name:
                    self.unshuffle_explorer.mapper.add_obstacle_in_front_of_agent(act_name = act_name,rotation = self.unshuffle_explorer.rotation,held_mode = held_mode)
                if args.debug_print:
                    print("ACTION FAILED (rearrange).", self.unshuffle_explorer.controller.last_event.metadata['errorMessage'])
                    print("message",self.unshuffle_explorer.controller.last_event.metadata["errorMessage"])
                self.rearrange_reset_execution_and_acts(
                    obj_pos=obj_pos, held_mode=held_mode,act_id=act_id)

            step += 1
            
        return prev_act_id
    
    def get_pick_up_obj_id_by_vis_feature(self,label,obj_feature):
        object_type = CATEGORY_LIST[label - 1]
        rgb_prev = self.unshuffle_explorer.rgb
        seg_prev, segmented_dict = self.unshuffle_explorer.segHelper.get_seg_pred(rgb=rgb_prev,isPickOrPutStage=True )
        if args.debug_print:
            print("segmented_dict categories", list(segmented_dict.keys()))
    
        if object_type in segmented_dict.keys():
            curr_in_view_seg_possible_objects = segmented_dict[object_type] 
            choose_mask = None
            max_feature_similarity = 0
            for seg_ins_id, possible_instance in curr_in_view_seg_possible_objects.items():
                possible_feature_similarity = cosine_similarity(possible_instance['feature'], obj_feature)
                if  possible_feature_similarity > max_feature_similarity:
                    max_feature_similarity = possible_feature_similarity
                    choose_mask = possible_instance['mask']
            
            assert choose_mask is not None
            
            simulator_provide_instance_id = None
            max_iou = 0
            for simulator_instance_id, simulator_mask in self.unshuffle_explorer.controller.last_event.instance_masks.items():
                simulator_category = simulator_instance_id.split('|')[0]
                if simulator_category not in self.rearrange_pickupable:
                    continue
                iou = calculate_iou_numpy(simulator_mask, choose_mask)
                if iou > max_iou:
                    max_iou = iou
                    simulator_provide_instance_id = simulator_instance_id
                
            object_id = simulator_provide_instance_id
            return object_id
        return None
    
    def get_pick_up_obj_id_by_pointcloud(self,point_list):
        depth = self.unshuffle_explorer.depth
        point_list = [(row[0],row[1],row[2]) for row in point_list]

        XYZ4 = np.trunc(XYZ3 * 100) / 100

        intersect_objs_id = None
        max_intersect_points_num = 0

        for objectId, mask in self.unshuffle_explorer.controller.last_event.instance_masks.items():
            object_indices = np.where(mask == 1)
    

            object_point_cloud = [(row[0],row[1],row[2]) for row in XYZ4[object_indices]]
            intersection = list(set(point_list) & set(object_point_cloud))
            if len(intersection) > max_intersect_points_num:
                intersect_objs_id = objectId
                max_intersect_points_num = len(intersection)
        
        return intersect_objs_id
            

        # tree = cKDTree(point_list)
    
        # distances, indices = tree.query(XYZ4.reshape(-1, 3), distance_upper_bound=1e-5)
        
        # matching_indices = np.where(distances == 0)[0]
    
        # matching_pixels = np.column_stack(np.unravel_index(matching_indices, (480, 480))).astype(float)

        # height,width = depth.shape
        # matching_pixels[:,0] = matching_pixels[:,0] / height
        # matching_pixels[:,1] = matching_pixels[:,1] / width

        # objects_id_list = []

        # for pixel in matching_pixels:
        #     query = self.unshuffle_explorer.controller.step(
        #         action="GetObjectInFrame",
        #         x=pixel[1],
        #         y=pixel[0],
        #         checkVisible=False
        #     )
        #     object_id = query.metadata["actionReturn"]
        #     objects_id_list.append(object_id)
        
        # if len(objects_id_list) == 0:
        #     return None

        # count = Counter(objects_id_list)
        # return count.most_common(1)[0][0]



    def pick_up_object(self, obj_centroid, ins_id):
        print("Preparing pick up object")
        # self.rotate_to_goal(obj_centroid)
        obj_id = self.get_pick_up_obj_id_by_pointcloud(self.unshuffle_instance_point_cloud_list[ins_id]["points"])
        if obj_id is not None:
            act_id = PICKUP
            paras = {"objectId": obj_id}
            print("Preparing pick up ",obj_id)
            self.act(PICKUP,paras)

            metadata = self.unshuffle_explorer.controller.last_event.metadata
            if self.unshuffle_explorer.controller.last_event.metadata["lastActionSuccess"] and len(metadata["inventoryObjects"]):
                if self.unshuffle_explorer.vis != None:
                    if args.use_seg:
                        seg_prev, segmented_dict = self.unshuffle_explorer.segHelper.get_seg_pred(
                            self.unshuffle_explorer.rgb, isPickOrPutStage = True)
                        seg_prev_vis = visualize_segmentationRGB(
                            self.unshuffle_explorer.rgb, segmented_dict, visualize_sem_seg=True)
                    else:
                        seg_prev_vis = None
                    self.unshuffle_explorer.vis.add_frame(image=self.unshuffle_explorer.rgb, seg_image=seg_prev_vis, add_map=True, 
                                                          add_map_seg=True, mapper=self.unshuffle_explorer.mapper, selem_agent_radius=self.unshuffle_explorer.selem_agent_radius,
                                                          step = self.unshuffle_explorer.step_from_stage_start, stage = self.unshuffle_explorer.curr_stage,
                                                          selem=self.unshuffle_explorer.selem, action="Hide_Object",text = self.error_message)
                
                # self.unshuffle_explorer.controller.step('HideObject',objectId=obj_id)
                self.unshuffle_explorer.rgb = self.unshuffle_explorer.controller.last_event.frame
                self.unshuffle_explorer.depth = self.unshuffle_explorer.controller.last_event.depth_frame
                self.error_message = self.unshuffle_explorer.controller.last_event.metadata[
                    'errorMessage']
                print("pick up "+ "success !")
                return True
            elif len(metadata["inventoryObjects"]) == 0 and self.unshuffle_explorer.controller.last_event.metadata["lastActionSuccess"]:
                print("error:PickupObject` was successful in picking up " + " but we're not holding any objects")
                return False
            else:
                print("pick up failed !")
                return False
        else:
            print("object is not seen, pick up failed")
            return False

        
    def rotate_to_goal(self,goal_centroid):
        agent_xy = np.array([self.unshuffle_explorer.position['x'], self.unshuffle_explorer.position['z']], np.float32)
        agent_xy = agent_xy - self.unshuffle_explorer.mapper.origin_xz + self.unshuffle_explorer.mapper.origin_map * self.unshuffle_explorer.mapper.resolution
        goal_xy = np.array([goal_centroid[0],goal_centroid[1]],np.float32)

        dx = goal_xy[0] - agent_xy[0]
        dy = goal_xy[1] - agent_xy[1]
        failed_num = 0

        act_seq = deque()
        if dy >= 0 and abs(dx) <= dy:
            if self.unshuffle_explorer.rotation == 90:
                act_seq.append(ROTATE_LEFT)
            elif self.unshuffle_explorer.rotation == 180:
                act_seq.append(ROTATE_LEFT)
                act_seq.append(ROTATE_LEFT)
            elif self.unshuffle_explorer.rotation == 270:
                act_seq.append(ROTATE_RIGHT)
        elif dx >= 0 and abs(dy) <= dx:
            if self.unshuffle_explorer.rotation == 180:
                act_seq.append(ROTATE_LEFT)
            elif self.unshuffle_explorer.rotation == 270:
                act_seq.append(ROTATE_LEFT)
                act_seq.append(ROTATE_LEFT)
            elif self.unshuffle_explorer.rotation == 0:
                act_seq.append(ROTATE_RIGHT)
        elif dy <= 0 and abs(dx) <= abs(dy):
            if self.unshuffle_explorer.rotation == 270:
                act_seq.append(ROTATE_LEFT)
            elif self.unshuffle_explorer.rotation == 0:
                act_seq.append(ROTATE_LEFT)
                act_seq.append(ROTATE_LEFT)
            elif self.unshuffle_explorer.rotation == 90:
                act_seq.append(ROTATE_RIGHT)
        elif dx <= 0 and abs(dy) <= abs(dx):
            if self.unshuffle_explorer.rotation == 0:
                act_seq.append(ROTATE_LEFT)
            elif self.unshuffle_explorer.rotation == 90:
                act_seq.append(ROTATE_LEFT)
                act_seq.append(ROTATE_LEFT)
            elif self.unshuffle_explorer.rotation == 180:
                act_seq.append(ROTATE_RIGHT)
        while bool(act_seq) and failed_num < 3:
            act_id = act_seq.pop()
            act_isSuccess = self.act(act_id=act_id)
            if not act_isSuccess:
                failed_num +=1
                if act_id == ROTATE_LEFT:
                    act_seq.append(ROTATE_LEFT)
                    act_seq.append(MOVE_RIGHT)
                elif act_id == ROTATE_RIGHT:
                    act_seq.append(ROTATE_RIGHT)
                    act_seq.append(MOVE_LEFT)
                else:
                    return
        act_seq = deque()
        failed_num = 0
        agent_xy = agent_xy - self.unshuffle_explorer.mapper.origin_xz + self.unshuffle_explorer.mapper.origin_map * self.unshuffle_explorer.mapper.resolution
        dist = np.sqrt(np.sum(np.square(agent_xy - goal_xy)))
        delta_height =  self.unshuffle_explorer.position['y'] - goal_centroid[2]#agent_height - object_hieght
        if abs(delta_height) > dist:
            if delta_height > 0:
                if self.unshuffle_explorer.head_tilt == 0:
                    act_seq.append(LOOK_DOWN)
                elif self.unshuffle_explorer.head_tilt == -30:
                    act_seq.append(LOOK_DOWN)
                    act_seq.append(LOOK_DOWN)
                elif self.unshuffle_explorer.head_tilt == 60:
                    act_seq.append(LOOK_UP)
            elif delta_height < 0:
                if self.unshuffle_explorer.head_tilt == 0:
                    act_seq.append(LOOK_UP)
                elif self.unshuffle_explorer.head_tilt == 30:
                    act_seq.append(LOOK_UP)
                    act_seq.append(LOOK_UP)
                elif self.unshuffle_explorer.head_tilt == 60:
                    act_seq.append(LOOK_UP)
                    act_seq.append(LOOK_UP)
                    act_seq.append(LOOK_UP)
        else:
            if self.unshuffle_explorer.head_tilt == 30:
                act_seq.append(LOOK_UP)
            elif self.unshuffle_explorer.head_tilt == 60:
                act_seq.append(LOOK_UP)
                act_seq.append(LOOK_UP)
            elif self.unshuffle_explorer.head_tilt == -30:
                act_seq.append(LOOK_DOWN)
        while bool(act_seq) and failed_num < 3:
            act_id = act_seq.pop()
            act_isSuccess = self.act(act_id=act_id)
            if not act_isSuccess:
                failed_num +=1
                if act_id == LOOK_UP or act_id == LOOK_DOWN:
                    act_seq.append(act_id)
                    act_seq.append(MOVE_BACK)
                else:
                    return
                
    def put_object(self,goal_centroid):
        print("preparing put object")

        self.rotate_to_goal(goal_centroid=goal_centroid)

        if self.unshuffle_explorer.vis != None:
            if args.use_seg:
                seg_prev, segmented_dict = self.unshuffle_explorer.segHelper.get_seg_pred(
                    rgb=self.unshuffle_explorer.rgb, isPickOrPutStage = True)
                seg_prev_vis = visualize_segmentationRGB(
                    self.unshuffle_explorer.rgb, segmented_dict, visualize_sem_seg=True)
            else:
                seg_prev_vis = None
            self.unshuffle_explorer.vis.add_frame(image=self.unshuffle_explorer.rgb, seg_image=seg_prev_vis, add_map=True, 
                                                  add_map_seg=True, mapper=self.unshuffle_explorer.mapper, selem_agent_radius=self.unshuffle_explorer.selem_agent_radius,
                                                  step = self.unshuffle_explorer.step_from_stage_start, stage = self.unshuffle_explorer.curr_stage,
                                                  selem=self.unshuffle_explorer.selem, action="Unhide",text = self.error_message)
        obj_id = self.held_object["objectId"]
        # self.unshuffle_explorer.controller.step('UnhideObject',objectId=self.held_object["objectId"])
        self.unshuffle_explorer.rgb = self.unshuffle_explorer.controller.last_event.frame
        self.unshuffle_explorer.depth = self.unshuffle_explorer.controller.last_event.depth_frame
        self.error_message = self.unshuffle_explorer.controller.last_event.metadata[
            'errorMessage']

        
        if self.unshuffle_explorer.vis != None:
            if args.use_seg:
                seg_prev, segmented_dict = self.unshuffle_explorer.segHelper.get_seg_pred(
                    rgb=self.unshuffle_explorer.rgb, isPickOrPutStage=True)
                seg_prev_vis = visualize_segmentationRGB(
                    self.unshuffle_explorer.rgb, segmented_dict, visualize_sem_seg=True)
            else:
                seg_prev_vis = None
            self.unshuffle_explorer.vis.add_frame(image=self.unshuffle_explorer.rgb, seg_image=seg_prev_vis, 
                                                    add_map=True, add_map_seg=True, mapper=self.unshuffle_explorer.mapper, selem_agent_radius=self.unshuffle_explorer.selem_agent_radius,
                                                    step = self.unshuffle_explorer.step_from_stage_start, stage = self.unshuffle_explorer.curr_stage,
                                                    selem=self.unshuffle_explorer.selem, action="drop_held_object_with_snap",text = self.error_message)
        paras = {"objectId":self.held_object["objectId"]}
        self.unshuffle_explorer.rgb = self.unshuffle_explorer.controller.last_event.frame
        self.unshuffle_explorer.depth = self.unshuffle_explorer.controller.last_event.depth_frame
        self.error_message = self.unshuffle_explorer.controller.last_event.metadata[
            'errorMessage']
        self.unshuffle_explorer._increment_num_steps_taken()
        # act_isSuccess = self.unshuffle_explorer.check_successful_action() == True
        self.unshuffle_explorer.controller.step('UnhideObject',objectId=obj_id)
        self.unshuffle_explorer._update_metrics(
            action_name='DropObject_snap', action_success=drop_success,paras = paras)
        if drop_success:
            return
            
        if self.held_object is not None:
            self.put_or_drop_object()
            

    @property
    def held_object(self) -> Optional[Dict[str, Any]]:
        """Return the data corresponding to the object held by the agent (if
        any)."""
        metadata = self.unshuffle_explorer.controller.last_event.metadata
        if len(metadata["inventoryObjects"]) == 0:
            return None
        assert len(metadata["inventoryObjects"]) <= 1
        held_obj_id = metadata["inventoryObjects"][0]["objectId"]
        return next(o for o in metadata["objects"] if o["objectId"] == held_obj_id)

    def drop_held_object_with_snap(self) -> bool:
        """Drop the object in the agent's hand to the target position.

        Exception is raised if shuffle has not yet been called on the current
        episode or the agent is in default mode.

        For this action to work:
            1. The agent must be within 1.5 meters from the goal object's
               position, observed during the walkthrough phase.
            2. The agent must be looking in the direction of where it was
               located in the walkthrough phase.

        Otherwise, the object will be placed in a visible receptacle or
        if this also fails, it will be simply dropped.

        # Returns

        `True` if the drop was successful, otherwise `False`.
        """

        # round positions to 2 decimals
        DEC = 2

        event = self.unshuffle_explorer.controller.last_event
        held_obj = self.held_object
        if held_obj is None:
            return False
        # When dropping up an object, make it breakable.
        self.unshuffle_explorer.controller.step(
            "MakeObjectBreakable", objectId=self.held_object["objectId"]
        )
        agent = event.metadata["agent"]
        goal_pose = self.walkthrough_objs_id_to_pose[held_obj["name"]]
        goal_pos = goal_pose["position"]
        goal_rot = goal_pose["rotation"]
        # good_positions_to_drop_from = self._interactable_positions_cache.get(
        #     scene_name=self.task['unique_id'] + '_' + self.task['roomSpec'],
        #     obj={**held_obj, **{"position": goal_pos, "rotation": goal_rot}, },
        #     controller=self.unshuffle_explorer.controller,
        #     # Forcing cache resets when not training.
        #     force_cache_refresh=True,
        # )
        good_positions_to_drop_from = self._interactable_positions_cache.get(
            scene_name=self.task_id.split("_")[0],
            obj={**held_obj, **{"position": goal_pos, "rotation": goal_rot}, },
            controller=self.unshuffle_explorer.controller,
            # Forcing cache resets when not training.
            force_cache_refresh=True,
        )

        def position_to_tuple(position: Dict[str, float]):
            return tuple(round(position[k], DEC) for k in ["x", "y", "z"])

        agent_xyz = position_to_tuple(agent["position"])
        agent_rot = (round(agent["rotation"]["y"] / args.DT) * args.DT) % 360
        agent_standing = int(agent["isStanding"])
        agent_horizon = round(agent["cameraHorizon"])
        for valid_agent_pos in good_positions_to_drop_from:

            valid_xyz = position_to_tuple(valid_agent_pos)
            valid_rot = (
                round(valid_agent_pos["rotation"] / args.DT) * args.DT) % 360
            valid_standing = int(valid_agent_pos["standing"])
            valid_horizon = round(valid_agent_pos["horizon"])
            check = (valid_xyz == agent_xyz  # Position
                     and valid_rot == agent_rot  # Rotation
                     and valid_standing == agent_standing  # Standing
                     and round(valid_horizon) == agent_horizon)  # Horizon
            if check:
                # Try a few locations near the target for robustness' sake
                print("check success,agent pos is in good_positions_to_drop_from")
                positions = [
                    {
                        "x": goal_pos["x"] + 0.001 * xoff,
                        "y": goal_pos["y"] + 0.001 * yoff,
                        "z": goal_pos["z"] + 0.001 * zoff,
                    }
                    for xoff in [0, -1, 1]
                    for zoff in [0, -1, 1]
                    for yoff in [0, 1, 2]
                ]
                self.unshuffle_explorer.controller.step(
                    action="TeleportObject",
                    objectId=held_obj["objectId"],
                    rotation=goal_rot,
                    positions=positions,
                    forceKinematic=True,
                    allowTeleportOutOfHand=True,
                    makeUnbreakable=True,
                )
                print('TeleportObject', self.unshuffle_explorer.controller.last_event.metadata[
                      'lastActionSuccess'], self.unshuffle_explorer.controller.last_event.metadata['errorMessage'])
                break
        if self.held_object is None:
            cur_pose = next(
                o for o in self.unshuffle_explorer.controller.last_event.metadata['objects'] if o["name"] == held_obj["name"])
            if are_poses_equal(goal_pose=goal_pose, cur_pose=cur_pose, treat_broken_as_unequal=True):
                print("drop object with snap success")
                return True
            else:
                print(
                    "drop object with snap failed, it was placed into the wrong location")
                return True
        else:
            return False
        
    def open_object_with_snap(self,object_type):
        curr_poses = self.unshuffle_explorer.get_current_objs_id_to_pose()
        obj_name_to_goal_and_cur_poses = {}
        action_success = False
        with include_object_data(self.unshuffle_explorer.controller):
            for obj_name in curr_poses.keys():
                obj_name_to_goal_and_cur_poses[obj_name] = (self.walkthrough_objs_id_to_pose[obj_name],curr_poses[obj_name])
            
            goal_pose = None
            cur_pose = None
            for o in self.unshuffle_explorer.controller.last_event.metadata["objects"]:
                if (
                    o["visible"]
                    and o["objectType"] == object_type
                    and o["openable"]
                    and not are_poses_equal(*obj_name_to_goal_and_cur_poses[o["name"]],treat_broken_as_unequal=True)
                ):
                    goal_pose, cur_pose = obj_name_to_goal_and_cur_poses[o["name"]]
                    break
            if goal_pose is not None:
                object_id = cur_pose["objectId"] 
                goal_openness = goal_pose["openness"]
                if cur_pose["openness"] > 0.0:
                    self.unshuffle_explorer.controller.step(
                        "CloseObject",
                        objectId=object_id,
                    )
                self.unshuffle_explorer.controller.step(
                    "OpenObject",
                    objectId=object_id,
                    openness=goal_openness,
                )
                action_success = self.unshuffle_explorer.controller.last_event.metadata[
                    "lastActionSuccess"
                ]
                if not action_success:
                    print(f"open {object_id} with snap failed",self.unshuffle_explorer.controller.last_event.metadata["errorMessage"])
                else:
                    print(f"open {object_id} with snap success")
            else:
                action_success = False
                print(f"goal object type {object_type} is not found")
        
        return action_success
    
    def act(self,act_id, paras=dict(),path=None,point_goal = None):
        rgb_prev = self.unshuffle_explorer.rgb
        depth_prev = self.unshuffle_explorer.depth
        if self.unshuffle_explorer.vis != None:
            if args.use_seg:
                seg_prev, segmented_dict = self.unshuffle_explorer.segHelper.get_seg_pred(
                    rgb=rgb_prev)
                seg_prev_vis = visualize_segmentationRGB(
                    rgb_prev, segmented_dict, visualize_sem_seg=True)
            else:
                seg_prev_vis = None
            if act_id == OPEN or act_id == PICKUP:
                self.unshuffle_explorer.vis.add_frame(image=rgb_prev, seg_image=seg_prev_vis, add_map=True, add_map_seg=True, 
                                                  point_goal=point_goal,path = path,
                                                  mapper=self.unshuffle_explorer.mapper, selem=self.unshuffle_explorer.selem, selem_agent_radius=self.unshuffle_explorer.selem_agent_radius,
                                                  step = self.unshuffle_explorer.step_from_stage_start, stage = self.unshuffle_explorer.curr_stage,
                                                  action=f"{self.unshuffle_explorer.act_id_to_name[act_id]}_{paras.values()}",text = self.error_message)
            else:
                self.unshuffle_explorer.vis.add_frame(image=rgb_prev, seg_image=seg_prev_vis, add_map=True, add_map_seg=True, 
                                                  point_goal=point_goal,path = path,
                                                  mapper=self.unshuffle_explorer.mapper, selem=self.unshuffle_explorer.selem, selem_agent_radius=self.unshuffle_explorer.selem_agent_radius,
                                                  step = self.unshuffle_explorer.step_from_stage_start, stage = self.unshuffle_explorer.curr_stage,
                                                  action=self.unshuffle_explorer.act_id_to_name[act_id],text = self.error_message)
        
        act_name = self.unshuffle_explorer.act_id_to_name[act_id]
        act_isSuccess = False
        if act_name.startswith("Open"):
            act_isSuccess = self.open_object_with_snap(paras["object_type"])
        else:
            self.unshuffle_explorer.controller.step(act_name,**paras)
            act_isSuccess = self.unshuffle_explorer.check_successful_action() == True
        if act_name.startswith("Pick"):
            self.unshuffle_explorer.controller.step('HideObject',objectId=paras["objectId"])
        self.unshuffle_explorer.rgb = self.unshuffle_explorer.controller.last_event.frame
        self.unshuffle_explorer.depth = self.unshuffle_explorer.controller.last_event.depth_frame
        self.error_message = self.unshuffle_explorer.controller.last_event.metadata[
            'errorMessage']
        self.unshuffle_explorer._increment_num_steps_taken()

        self.unshuffle_explorer._update_metrics(
            action_name=act_name, action_success=act_isSuccess,paras = paras)
        if act_isSuccess:
            self.unshuffle_explorer.update_position_and_rotation(act_id)
            self.unshuffle_explorer.mapper.update_position_on_map(
                self.unshuffle_explorer.position, self.unshuffle_explorer.rotation)
        return act_isSuccess

    def put_or_drop_object(self):
        for i in range(4):
            # We couldn't teleport the object to the target location, let's try placing it
            # in a visible receptacle.
            possible_receptacles = [
                o for o in self.unshuffle_explorer.controller.last_event.metadata["objects"] if o["visible"] and o["receptacle"]
            ]
            possible_receptacles = sorted(
                possible_receptacles, key=lambda o: (o["distance"], o["objectId"])
            )
            for possible_receptacle in possible_receptacles:
                paras = {"objectId":possible_receptacle["objectId"]}
                self.act(PUT,paras)
                if self.unshuffle_explorer.controller.last_event.metadata["lastActionSuccess"]:
                    print(
                        "Can't teleport the object to the target location, try placing it in a visible receptacle")
                    return
            if self.unshuffle_explorer.head_tilt == 0:
                look_down_suc = self.act(LOOK_DOWN)
                if look_down_suc:
                    possible_receptacles = [
                        o for o in self.unshuffle_explorer.controller.last_event.metadata["objects"] if o["visible"] and o["receptacle"]
                    ]
                    possible_receptacles = sorted(
                        possible_receptacles, key=lambda o: (o["distance"], o["objectId"])
                    )
                    for possible_receptacle in possible_receptacles:
                        paras = {"objectId":possible_receptacle["objectId"]}
                        self.act(PUT,paras)
                        if self.unshuffle_explorer.controller.last_event.metadata["lastActionSuccess"]:
                            print(
                                "Can't teleport the object to the target location, try placing it in a visible receptacle")
                            return
            rotate_suc = self.act(ROTATE_LEFT)
            if not rotate_suc:
                break
            
        # We failed to place the object into a receptacle, let's just drop it.
        if self.held_object is not None:
            paras = {"forceAction":True}
            self.act(DROP,paras)
            print("failed to teleport the object to the target location or place the object into a receptacle, just drop it")

    def move_object_state(self, cur_state, goal_state, cur_ins_id,step_max,cur_centorid=None,goal_centroid=None):
        
        print("Preparing to rearrange object from ", cur_state,
              " to ", goal_state, " with each navigation in ", step_max, " steps!")
        last_action = self.navigate_to_object_pos(obj_pos=cur_state, step_max=step_max, held_mode=False)
        if last_action == DONE:
            obj_height = cur_state[2]
            self.rotate_to_goal(cur_centorid)
            for i in range(2):
                pick_up_success  = self.pick_up_object(cur_centorid,cur_ins_id)
                print("!!!!!!!!!!",i)
                if pick_up_success:
                    break
                else:
                    self.act(MOVE_BACK)
            if pick_up_success == False:
                return
        else:
            return
        last_action = self.navigate_to_object_pos(obj_pos=goal_state, step_max=step_max, held_mode=True)
        self.put_object(goal_centroid)

    def match_two_relations(self):
        walkthrough_obj_rels = self.relation_calculator.get_relations_pickupable(self.walkthrough_explorer.object_tracker)
        unshuffle_obj_rels = self.relation_calculator.get_relations_pickupable(self.unshuffle_explorer.object_tracker)
        if args.debug_print:
            print("walkthrough_obj_rels", walkthrough_obj_rels)
            print("unshuffle_obj_rels", unshuffle_obj_rels)
        for key in list(walkthrough_obj_rels.keys()):
            if key in unshuffle_obj_rels:
                count_similar = 0
                count_dissimilar = 0
                rels_u = unshuffle_obj_rels[key]['relations'][0]
                rels_w = walkthrough_obj_rels[key]['relations'][0]

                similar = []
                different = []
                for rel1 in rels_u:
                    found_one = False
                    for rel2 in rels_w:
                        if rel1 == rel2:
                            count_similar += 1
                            found_one = True
                            similar.append(rel1)
                            break
                    if not found_one:
                        count_dissimilar += 1
                        different.append(rel1)
                print("OBJECT", key)
                print("Num similar=", count_similar,
                      "Num dissimilar=", count_dissimilar)
                count_dissimilar += 1e-6
                oop = count_similar / \
                    count_dissimilar < args.dissimilar_threshold and count_dissimilar > args.thresh_num_dissimilar

                if oop:
                    obj_info = {}
                    obj_info["label"] = CATEGORY_to_ID[key]

                    centroid = unshuffle_obj_rels[key]['centroids'][0]
                    x_bin = np.round(centroid[0] / args.map_resolution).astype(np.int32)
                    y_bin = np.round(centroid[1] / args.map_resolution).astype(np.int32)
                    z_bin = np.digitize(centroid[2], bins=self.unshuffle_explorer.z).astype(np.int32)
                    obj_info["cur_map_pos"] = np.array([y_bin, x_bin, z_bin])

                    centroid = walkthrough_obj_rels[key]['centroids'][0]
                    x_bin = np.round(centroid[0] / args.map_resolution).astype(np.int32)
                    y_bin = np.round(centroid[1] / args.map_resolution).astype(np.int32)
                    z_bin = np.digitize(centroid[2], bins=self.unshuffle_explorer.z).astype(np.int32)
                    obj_info["origin_map_pos"] = np.array([y_bin, x_bin, z_bin])

                    self.object_to_rearrange_dict_list.append(obj_info)

    def match_feature_then_relation(self):
        walkthrough_object_track_dict = self.walkthrough_explorer.object_tracker.objects_track_dict
        unshuffle_object_track_dict = self.unshuffle_explorer.object_tracker.objects_track_dict

        walkthrough_obj_rels = self.relation_calculator.get_relations_pickupable(self.walkthrough_explorer.object_tracker)
        unshuffle_obj_rels = self.relation_calculator.get_relations_pickupable(self.unshuffle_explorer.object_tracker)

        for unshuffle_category, unshuffle_instances in unshuffle_object_track_dict.items():
            category_id = CATEGORY_to_ID[unshuffle_category]
            if CATEGORY_LIST[category_id-1] not in self.rearrange_pickupable:
                continue
    
            if unshuffle_category not in walkthrough_object_track_dict:
                print(f'unshuffle found {len(unshuffle_instances)} {unshuffle_category} while not found in the walkthrough stage')
                continue 
            G = nx.Graph()
            edges = []
            walkthrough_instances = walkthrough_object_track_dict[unshuffle_category]
            for unshuffle_ins_id, unshuffle_ins in unshuffle_instances.items():
                for walkthrough_ins_id, walkthrough_ins in walkthrough_instances.items():
                    source_node_key = str('unshuffle') + '_' + str(unshuffle_ins_id)
                    target_node_key = str('walkthrough') + '_' + str(walkthrough_ins_id)
                    instance_feature_similarity = cosine_similarity(unshuffle_ins['feature'],  walkthrough_ins['feature'])
                    if instance_feature_similarity < INSTANCE_FEATURE_SIMILARITY_THRESHOLD:
                        continue
                    else:
                        edges.append((source_node_key, target_node_key, instance_feature_similarity))
            if args.debug_print:
                print('edges: ', edges)         
            G.add_weighted_edges_from(edges)
            matching = nx.max_weight_matching(G)

            for matched_one_ins, matched_two_ins in matching:
                if "unshuffle" in matched_one_ins:
                    unshuffle_ins_id = int(matched_one_ins.split('_')[-1])
                    walkthrough_ins_id = int(matched_two_ins.split('_')[-1])
                else:
                    unshuffle_ins_id = int(matched_two_ins.split('_')[-1])
                    walkthrough_ins_id = int(matched_one_ins.split('_')[-1])
                unshuffle_ins_centroid = unshuffle_object_track_dict[unshuffle_category][str(unshuffle_category) + '_' + str(unshuffle_ins_id)]['centroid']
                walkthrought_ins_centroid  = walkthrough_object_track_dict[unshuffle_category][str(unshuffle_category) + '_' + str(walkthrough_ins_id)]['centroid']

                count_similar = 0
                count_dissimilar = 0
                rels_u = unshuffle_obj_rels[unshuffle_category][str(unshuffle_category) + '_' + str(unshuffle_ins_id)]['relations']
                rels_w = walkthrough_obj_rels[unshuffle_category][str(unshuffle_category) + '_' + str(walkthrough_ins_id)]['relations']

                similar = []
                different = []
                for rel1 in rels_u:
                    found_one = False
                    for rel2 in rels_w:
                        if rel1 == rel2:
                            count_similar += 1
                            found_one = True
                            similar.append(rel1)
                            break
                    if not found_one:
                        count_dissimilar += 1
                        different.append(rel1)
                print("Matched Instances: ", matched_one_ins, matched_two_ins)
                print("Num similar=", count_similar,
                      "Num dissimilar=", count_dissimilar)
                count_dissimilar += 1e-6
                oop = count_similar / count_dissimilar < args.dissimilar_threshold and count_dissimilar > args.thresh_num_dissimilar

                if oop:
                    self.object_to_rearrange_dict_list.append(
                        {
                            "label": category_id,
                            "cur_map_pos": np.array(unshuffle_ins_centroid),
                            "origin_map_pos": np.array(walkthrought_ins_centroid),
                            "cur_ins_id": unshuffle_ins_id,
                            "origin_ins_id": walkthrough_ins_id,
                        }
                    )


    def match_feature_then_centroid(self):
        walkthrough_object_track_dict = self.walkthrough_explorer.object_tracker.objects_track_dict
        unshuffle_object_track_dict = self.unshuffle_explorer.object_tracker.objects_track_dict
        for unshuffle_category, unshuffle_instances in unshuffle_object_track_dict.items():
            category_id = CATEGORY_to_ID[unshuffle_category]
            if CATEGORY_LIST[category_id-1] not in self.rearrange_pickupable:
                continue
    
            if unshuffle_category not in walkthrough_object_track_dict:
                print(f'unshuffle found {len(unshuffle_instances)} {unshuffle_category} while not found in the walkthrough stage')
                continue 
            G = nx.Graph()
            edges = []
            walkthrough_instances = walkthrough_object_track_dict[unshuffle_category]
            for unshuffle_ins_id, unshuffle_ins in unshuffle_instances.items():
                for walkthrough_ins_id, walkthrough_ins in walkthrough_instances.items():
                    source_node_key = str('unshuffle') + '_' + str(unshuffle_ins_id)
                    target_node_key = str('walkthrough') + '_' + str(walkthrough_ins_id)
                    instance_feature_similarity = cosine_similarity(unshuffle_ins['feature'],  walkthrough_ins['feature'])
                    if instance_feature_similarity < INSTANCE_FEATURE_SIMILARITY_THRESHOLD:
                        continue
                    else:
                        edges.append((source_node_key, target_node_key, instance_feature_similarity))
            if args.debug_print:            
                print('edges: ', edges)
            G.add_weighted_edges_from(edges)
            matching = nx.max_weight_matching(G)

            for matched_one_ins, matched_two_ins in matching:
                if "unshuffle" in matched_one_ins:
                    unshuffle_ins_id = int(matched_one_ins.split('_')[-1])
                    walkthrough_ins_id = int(matched_two_ins.split('_')[-1])
                else:
                    unshuffle_ins_id = int(matched_two_ins.split('_')[-1])
                    walkthrough_ins_id = int(matched_one_ins.split('_')[-1])
                unshuffle_ins_centroid = unshuffle_object_track_dict[unshuffle_category][str(unshuffle_category) + '_' + str(unshuffle_ins_id)]['centroid']
                walkthrought_ins_centroid  = walkthrough_object_track_dict[unshuffle_category][str(unshuffle_category) + '_' + str(walkthrough_ins_id)]['centroid']

                move_distance = np.linalg.norm(unshuffle_ins_centroid - walkthrought_ins_centroid)
                if move_distance > INSTANCE_CENTROID_THRESHOLD: 
                    self.object_to_rearrange_dict_list.append(
                        {
                            "label": category_id,
                            "cur_map_pos": np.array(unshuffle_ins_centroid),
                            "origin_map_pos": np.array(walkthrought_ins_centroid),
                            "cur_ins_id": unshuffle_ins_id,
                            "origin_ins_id": walkthrough_ins_id,
                            'move_distance': move_distance,
                        }
                    )



    def match_feature_then_IoU(self):
        walkthrough_object_track_dict = self.walkthrough_explorer.object_tracker.objects_track_dict
        unshuffle_object_track_dict = self.unshuffle_explorer.object_tracker.objects_track_dict
        for unshuffle_category, unshuffle_instances in unshuffle_object_track_dict.items():
            category_id = CATEGORY_to_ID[unshuffle_category]
            if CATEGORY_LIST[category_id-1] not in self.rearrange_pickupable:
                continue
    
            if unshuffle_category not in walkthrough_object_track_dict:
                print(f'unshuffle found {len(unshuffle_instances)} {unshuffle_category} while not found in the walkthrough stage')
                continue 
            G = nx.Graph()
            edges = []
            walkthrough_instances = walkthrough_object_track_dict[unshuffle_category]
            for unshuffle_ins_id, unshuffle_ins in unshuffle_instances.items():
                for walkthrough_ins_id, walkthrough_ins in walkthrough_instances.items():
                    source_node_key = str('unshuffle') + '_' + str(unshuffle_ins_id)
                    target_node_key = str('walkthrough') + '_' + str(walkthrough_ins_id)
                    instance_feature_similarity = cosine_similarity(unshuffle_ins['feature'],  walkthrough_ins['feature'])
                    if instance_feature_similarity < INSTANCE_FEATURE_SIMILARITY_THRESHOLD:
                        continue
                    else:
                        edges.append((source_node_key, target_node_key, instance_feature_similarity))
            if args.debug_print:
                print('edges: ', edges)
            G.add_weighted_edges_from(edges)
            matching = nx.max_weight_matching(G)

            for matched_one_ins, matched_two_ins in matching:
                if "unshuffle" in matched_one_ins:
                    unshuffle_ins_id = int(matched_one_ins.split('_')[-1])
                    walkthrough_ins_id = int(matched_two_ins.split('_')[-1])
                else:
                    unshuffle_ins_id = int(matched_two_ins.split('_')[-1])
                    walkthrough_ins_id = int(matched_one_ins.split('_')[-1])

                unshuffle_ins_semantic_map = self.unshuffle_explorer.mapper.semantic_map[:,:,:, category_id] == unshuffle_ins_id
                walkthrough_ins_semantic_map = self.walkthrough_explorer.mapper.semantic_map[:,:,:, category_id] == walkthrough_ins_id

                iou = calculate_iou_numpy(unshuffle_ins_semantic_map, walkthrough_ins_semantic_map)
                    continue
                else:
                    unshuffle_ins_centroid = unshuffle_object_track_dict[unshuffle_category][str(unshuffle_category) + '_' + str(unshuffle_ins_id)]['centroid']
                    walkthrought_ins_centroid  = walkthrough_object_track_dict[unshuffle_category][str(unshuffle_category) + '_' + str(walkthrough_ins_id)]['centroid']
                    self.object_to_rearrange_dict_list.append(
                        {
                            "label": category_id,
                            "cur_map_pos": np.array(unshuffle_ins_centroid),
                            "origin_map_pos": np.array(walkthrought_ins_centroid),
                            "cur_ins_id": unshuffle_ins_id,
                            "origin_ins_id": walkthrough_ins_id,
                            'iou': iou,
                        }
                    )
    def get_open_objs(self):
        centroids,labels = self.unshuffle_explorer.object_tracker.get_centroids_and_labels()
        assert len(centroids) == len(labels)
        for i in range(len(labels)):
            resolution = self.unshuffle_explorer.mapper.resolution
            z_bins = self.unshuffle_explorer.mapper.z_bins
            centroid = centroids[i]
            map_pos = np.array([centroid[1]/resolution,centroid[0]/resolution,np.digitize(centroid[2], bins=z_bins)]).astype(np.int32)
            
            self.open_objects_list.append(
                    {
                        "label": labels[i],
                        "map_pos": map_pos,
                        "centroid":centroid
                    }
                )
    
    def match_feature_then_point_cloud(self):
        import pyvista as pv
        walkthrough_global_pointcloud = self.walkthrough_explorer.global_point_cloud.keys()
        # point_cloud = pv.PolyData(np.array(list(walkthrough_global_pointcloud)))
        # point_cloud.save("walkthrough_global_pointcloud.vtk")
        walkthrough_add = set(self.walkthrough_explorer.global_point_cloud.keys())-set(self.unshuffle_explorer.global_point_cloud.keys())
        walkthrough_point_cloud = set(self.unshuffle_explorer.different_point_cloud["walkthrough"].keys())
        walkthrough_changed_point_cloud = list(walkthrough_add & walkthrough_point_cloud)
        # point_cloud = pv.PolyData(np.array(walkthrough_changed_point_cloud))
        # point_cloud.save("walkthrough_changed_pointcloud.vtk")

        unshuffle_global_pointcloud = self.unshuffle_explorer.global_point_cloud.keys()
        # point_cloud = pv.PolyData(np.array(list(unshuffle_global_pointcloud)))
        # point_cloud.save("unshuffle_global_pointcloud.vtk")
        unshuffle_add = set(self.unshuffle_explorer.global_point_cloud.keys())-set(self.walkthrough_explorer.global_point_cloud.keys())
        unshuffle_point_cloud = set(self.unshuffle_explorer.different_point_cloud["unshuffle"].keys())
        unshuffle_changed_point_cloud = list(unshuffle_add & unshuffle_point_cloud)
        # point_cloud = pv.PolyData(np.array(unshuffle_changed_point_cloud))
        # point_cloud.save("unshuffle_changed_pointcloud.vtk")

        # walkthrough_point_cloud = np.array(list(self.unshuffle_explorer.different_point_cloud["walkthrough"].keys()))
        # unshuffle_point_cloud = np.array(list(self.unshuffle_explorer.different_point_cloud["unshuffle"].keys()))
        # walkthrough_point_cloud = walkthrough_changed_point_cloud
        # unshuffle_point_cloud = unshuffle_changed_point_cloud
        self.walkthrough_instance_point_cloud_list = self.point_cloud_seg(walkthrough_changed_point_cloud,self.unshuffle_explorer.different_point_cloud["walkthrough"])
        self.unshuffle_instance_point_cloud_list = self.point_cloud_seg(unshuffle_changed_point_cloud,self.unshuffle_explorer.different_point_cloud["unshuffle"])
        for obj in self.walkthrough_instance_point_cloud_list:
            instance_geometry_feature = self.pointnet2_helper.pointcloud_binary_cls(obj["points"])
            obj["geometry"] = instance_geometry_feature
        for obj in self.unshuffle_instance_point_cloud_list:
            instance_geometry_feature = self.pointnet2_helper.pointcloud_binary_cls(obj["points"])
            obj["geometry"] = instance_geometry_feature
        G = nx.Graph()
        edges = []
        
        for unshuffle_ins_id, unshuffle_ins in enumerate(self.unshuffle_instance_point_cloud_list):
            for walkthrough_ins_id, walkthrough_ins in enumerate(self.walkthrough_instance_point_cloud_list):
                source_node_key = str('unshuffle') + '_' + str(unshuffle_ins_id)
                target_node_key = str('walkthrough') + '_' + str(walkthrough_ins_id)
                instance_feature_similarity = cosine_similarity(unshuffle_ins['feature'],  walkthrough_ins['feature'])
                # instance_geometry_similarity = self.pointnet2_helper.pointcloud_binary_cls(unshuffle_ins["points"],walkthrough_ins["points"])
                instance_geometry_similarity = cosine_similarity(unshuffle_ins["geometry"],walkthrough_ins["geometry"])
                instance_similarity = instance_feature_similarity + instance_geometry_similarity
                # instance_similarity = 1
                # if instance_feature_similarity < INSTANCE_FEATURE_SIMILARITY_THRESHOLD:
                #     continue
                # else:
                edges.append((source_node_key, target_node_key, instance_similarity))
        
        G.add_weighted_edges_from(edges)
        matching = nx.max_weight_matching(G)
        for matched_one_ins, matched_two_ins in matching:
            if "unshuffle" in matched_one_ins:
                unshuffle_ins_id = int(matched_one_ins.split('_')[-1])
                walkthrough_ins_id = int(matched_two_ins.split('_')[-1])
            else:
                unshuffle_ins_id = int(matched_two_ins.split('_')[-1])
                walkthrough_ins_id = int(matched_one_ins.split('_')[-1])

            walkthrough_points = self.walkthrough_instance_point_cloud_list[walkthrough_ins_id]["points"]
            unshuffle_points = self.unshuffle_instance_point_cloud_list[unshuffle_ins_id]["points"]

            walkthrough_ins_centroid = np.array([np.median(walkthrough_points[:,0]),np.median(walkthrough_points[:,1]),np.median(walkthrough_points[:,2])])
            unshuffle_ins_centroid = np.array([np.median(unshuffle_points[:,0]),np.median(unshuffle_points[:,1]),np.median(unshuffle_points[:,2])])
            # walkthrough_ins_centroid = np.mean(self.walkthrough_instance_point_cloud_list[walkthrough_ins_id]["points"],axis=0)
            # unshuffle_ins_centroid = np.mean(self.unshuffle_instance_point_cloud_list[unshuffle_ins_id]["points"],axis=0)

            resolution = self.unshuffle_explorer.mapper.resolution
            z_bins = self.unshuffle_explorer.mapper.z_bins
            walkthrough_ins_map_pos = np.array([walkthrough_ins_centroid[1]/resolution,walkthrough_ins_centroid[0]/resolution,np.digitize(walkthrough_ins_centroid[2], bins=z_bins)]).astype(np.int32)
            unshuffle_ins_map_pos = np.array([unshuffle_ins_centroid[1]/resolution,unshuffle_ins_centroid[0]/resolution,np.digitize(unshuffle_ins_centroid[2], bins=z_bins)]).astype(np.int32)
            
            self.object_to_rearrange_dict_list.append(
                    {
                        "label": None,
                        "cur_map_pos": unshuffle_ins_map_pos,
                        "origin_map_pos": walkthrough_ins_map_pos,
                        "cur_ins_id": unshuffle_ins_id,
                        "origin_ins_id": walkthrough_ins_id,
                        "cur_centroid":unshuffle_ins_centroid,
                        "origin_centroid":walkthrough_ins_centroid
                    }
                )
    
    def point_cloud_seg(self,changed_point_cloud,point_cloud:dict()):
        '''
            point_cloud_divided_by_instance = [
                    {
                        "points": np.ndarray([n,3]), #x,y,z
                        "feature":np.ndarray([m])
                    },
                    ...,
                    {
                        "points": np.ndarray([n,3]), #x,y,z
                        "feature":np.ndarray([m])
                    }
            ]
                
        '''
        instance_point_cloud_list = [] 
        # xyz = np.array(list(point_cloud.keys()))
        xyz = np.array(changed_point_cloud)
        if xyz.shape[0] != 0:
            clustering = DBSCAN(eps=0.2, min_samples=3).fit(xyz)
            unique_clusters = np.unique(cluster_labels)
            for label in unique_clusters:
                if label == -1:
                    continue
                instance_point_cloud = {}
                instance_xyz = xyz[cluster_labels == label]
                if instance_xyz.shape[0] < 30:
                    continue
                feature_indices = np.array([item for t in instance_xyz if (t[0],t[1],t[2]) in point_cloud for item in point_cloud[(t[0],t[1],t[2])]]).astype(int)
                selected_features = np.array([self.unshuffle_explorer.feature_list[i] for i in feature_indices])
                fuse_feature = selected_features.mean(axis=0)
                instance_point_cloud["points"] = np.array(instance_xyz)
                instance_point_cloud["feature"] = fuse_feature
                instance_point_cloud_list.append(instance_point_cloud)
        return instance_point_cloud_list

    def open_one_object(self,label,map_pos,centroid,step_max):
        print(f"Preparing to open {label} at {map_pos} with navigation in {step_max} steps")
        last_action = self.navigate_to_object_pos(obj_pos=map_pos, step_max=step_max, held_mode=False)
        self.rotate_to_goal(centroid)
        paras = {"object_type": label}
        for i in range(3):
            open_success = self.act(OPEN,paras)
            if open_success:
                break;
            else:
                self.act(MOVE_BACK)
            

        

    def rearange_objects(self):
        # self.match_two_map_from_category(self.walkthrough_explorer.mapper.semantic_map, self.unshuffle_explorer.mapper.semantic_map, dist_thresh=10)
        if self.solution_config['unshuffle_match'] == 'feature_IoU_based':
            self.match_feature_then_IoU()
        elif self.solution_config['unshuffle_match'] == 'feature_centroid_based':
            self.match_feature_then_centroid()
        elif self.solution_config['unshuffle_match'] == 'feature_relation_based':
            self.match_feature_then_relation()
        elif self.solution_config["unshuffle_match"] == 'feature_pointcloud_based':
            self.match_feature_then_point_cloud()
            
        print("Objects need to be rearranged: ",self.object_to_rearrange_dict_list)
        
        if self.solution_config['unshuffle_reorder'] == 'or_tools':
            objects_sequence = self.or_tools_reorder()
        elif self.solution_config['unshuffle_reorder'] == 'greedy':
            objects_sequence = self.greedy_reorder()
        elif self.solution_config['unshuffle_reorder'] == 'random':
            objects_sequence = self.random_reorder()
        print("rearrange order policy:",self.solution_config['unshuffle_reorder'], ', objects_sequence after order:', objects_sequence)
        if not objects_sequence:
            # objects_sequence = [i for i in range(
            #     len(self.object_to_rearrange_dict_list))]
            objects_sequence = self.greedy_reorder()
        if len(objects_sequence):
            for i in objects_sequence:
                cur_state = self.object_to_rearrange_dict_list[i]['cur_map_pos']
                cur_ins_id = self.object_to_rearrange_dict_list[i]['cur_ins_id']
                goal_state = self.object_to_rearrange_dict_list[i]['origin_map_pos']
                cur_centroid = self.object_to_rearrange_dict_list[i]["cur_centroid"]
                goal_centroid = self.object_to_rearrange_dict_list[i]["origin_centroid"]
                self.move_object_state(cur_state=cur_state,goal_state=goal_state,cur_ins_id=cur_ins_id,step_max=args.rearrange_step_max,cur_centorid=cur_centroid,goal_centroid=goal_centroid)
        self.get_open_objs()
        for obj in self.open_objects_list:
            self.open_one_object(label=obj["label"],map_pos=obj["map_pos"],centroid=obj["centroid"],step_max=args.rearrange_step_max)
        if self.unshuffle_explorer.vis != None:
            if args.use_seg:
                seg_prev, segmented_dict = self.unshuffle_explorer.segHelper.get_seg_pred(
                    self.unshuffle_explorer.rgb)
                seg_prev_vis = visualize_segmentationRGB(
                    self.unshuffle_explorer.rgb, segmented_dict, visualize_sem_seg=True)
            else:
                seg_prev_vis = None
            self.unshuffle_explorer.vis.add_frame(image=self.unshuffle_explorer.rgb, depth=self.unshuffle_explorer.depth,seg_image=seg_prev_vis, 
                                                  add_map=True, add_map_seg=True, mapper=self.unshuffle_explorer.mapper, selem_agent_radius=self.unshuffle_explorer.selem_agent_radius,
                                                  step = self.unshuffle_explorer.step_from_stage_start, stage = self.unshuffle_explorer.curr_stage,
                                                  selem=self.unshuffle_explorer.selem, action="Done",text = self.error_message)

def main():
    rearrange = Rearrange()
    distance_matrix = np.array([[0, 205, 85, 229, 241, 63, 85, 229, 229],
                                [205, 0, 174, 60, 72, 228, 174, 60, 60],
                                [85, 174, 0, 198, 210, 132, 0, 198, 198],
                                [229, 60, 198, 0, 12, 252, 198, 24, 0],
                                [241, 72, 210, 12, 0, 264, 210, 12, 12],
                                [63, 228, 132, 252, 264, 0, 132, 252, 252],
                                [85, 174, 0, 198, 210, 132, 0, 198, 198],
                                [229, 60, 198, 24, 12, 252, 198, 0, 24],
                                [229, 60, 198, 0, 12, 252, 198, 24, 0]])

    constraints = np.array([[], [3], [], []])
    print("constraints", constraints)
    data = rearrange.or_tools_create_data_model(
        distance_matrix=distance_matrix, constraints=constraints)
    route = rearrange.or_tools_compute_rearrage_order(data)
    print(route)
    order = []
    for i in range(1, len(route)):
        if route[i] % 2 == 0:
            order.append(int(route[i] / 2 - 1))
    print(order)


if __name__ == '__main__':
    main()
