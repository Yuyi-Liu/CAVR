import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2
import matplotlib.gridspec as gridspec
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from rearrange_on_proc.arguments import args
import os
import torch
import pdb
import skimage.morphology
from rearrange_on_proc.mapper import Mapper
from rearrange_on_proc.planner import FMMPlanner
from rearrange_on_proc.constants import CATEGORY_PALETTE, OTHER_PALETTE, SEGMENTATION_CATEGORY_to_COLOR, CATEGORY_LIST
from PIL import Image
from rearrange_on_proc.utils.utils import pool2d, RGB_to_Hex, get_colormap
from rearrange_on_proc.utils.aithor import  get_origin_T_camX, get_3dbox_in_geom_format, get_amodal2d
import math
import imageio
from PIL import Image, ImageDraw
import copy

class Animation():
    '''
    util for generating movies of the agent and TIDEE modules
    '''

    def __init__(self, W,H, name_to_id=None):  

        self.fig = plt.figure(1, figsize=(30, 15))
        plt.clf()

        self.W = W
        self.H = H

        self.name_to_id = name_to_id

        self.object_tracker = None
        self.camX0_T_origin = None

        self.image_plots = []


    def add_frame(self, image:np.ndarray, seg_image, mapper:Mapper, add_map:bool, add_map_seg: bool, selem, selem_agent_radius, action,depth = None,point_goal:list = None,path=None,stage = None, step=None,visdom=None,text=None):
        # ncols = 2
        # plt.figure(1)
        # plt.clf()
        
        # ax = []
        # spec = gridspec.GridSpec(ncols=ncols, nrows=1, 
        #         figure=self.fig, left=0., right=1., wspace=0.05, hspace=0.5)
        # ax.append(self.fig.add_subplot(spec[0, 0]))
        # ax.append(self.fig.add_subplot(spec[0, 1]))
        # for a in ax:
        #     a.axis('off')
        # image = image.astype(np.uint8)
        # ax[0].imshow(image)
        # ax[1].imshow(image)
        
        
        plt.clf()
        ax = []
        step_pix = int(args.STEP_SIZE / args.map_resolution)

        
        ncols = 2
        spec = gridspec.GridSpec(ncols=2, nrows=1, 
                    figure=self.fig, left=0., right=1., wspace=0.0001, hspace=0.0001)

        # spec.tight_layout(figure = self.fig, w_pad=0, h_pad=0)
        ax.append(self.fig.add_subplot(spec[0, 0]))  # rgb
        ax.append(self.fig.add_subplot(spec[0, 1]))  # map
        # ax.append(self.fig.add_subplot(spec[0, 2])) # obstacle_map

        for a in ax:
            a.axis('off')
        if step is not None:
            ax[0].set_title(f"{stage} stage -- step:{step}: {action}")
        else:
            ax[0].set_title(f"{action}")

        if stage == "unshuffle":
            Image.fromarray(image).save(f"supplementary/rearrange_img/{step}_rgb.jpg")
        # image = image.astype(np.uint8)
        # seg_image = seg_image.astype(np.uint8)
        # ax[0].imshow(seg_image)
        
        # m_vis = np.invert(mapper.get_traversible_map(
        #     selem_agent_radius, 1,loc_on_map_traversible=False))
        # ax[1].imshow(m_vis, origin='lower', vmin=0, vmax=1,
        #         cmap='Greys')
        # state_xy = mapper.get_position_on_map()
        # state_theta = mapper.get_rotation_on_map()
        # arrow_len = 1.0/mapper.resolution
        # ax[1].arrow(state_xy[0], state_xy[1], 
        #             arrow_len*np.cos(state_theta+np.pi/2),
        #             arrow_len*np.sin(state_theta+np.pi/2), 
        #             color='b', head_width=10)
        
        # if point_goal is not None:
        #     ax[1].plot(point_goal[1], point_goal[0], color='blue', marker='o',linewidth=10, markersize=6)
        # # ax[3].imshow(m_vis, origin='lower', vmin=0, vmax=1,
        # #          cmap='Greys')
        # if path != None:
        #     for i in range(len(path)):
        #         x,y = path[i]
        #         x = x*step_pix
        #         y = y*step_pix
        #         ax[1].plot(y, x, 'ro',markersize = 4)
    
        # ax[1].plot(state_xy[0], state_xy[1], 'go',markersize = 8)
        # if point_goal is not None:
        #     ax[1].plot(point_goal[1], point_goal[0], 'bo',markersize = 8)
        # if text is not None:
        #     ax[1].set_title(text)
        # else:
        #     ax[1].set_title("path")
        
        # canvas = FigureCanvas(plt.gcf())

        # canvas.draw()       # draw the canvas, cache the renderer
        # width, height = plt.gcf().get_size_inches() * plt.gcf().get_dpi()
        # image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

        # # visdom.matplot(plt)        
        # # pdb.set_trace()
        # self.image_plots.append(image)

    def render_movie(self, dir,  process_id, episode, tag='', fps=5):

        if not os.path.exists(dir):
            os.mkdir(dir)
        video_name = os.path.join(dir, f'output-pid{process_id}-task{episode}|{tag}.mp4')
        print(f"rendering to {video_name}")
        height, width, _ = self.image_plots[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_name, fourcc, 4, (width, height))

        # print("------------------again__------------", len(self.image_plots))
        for im in self.image_plots:
            rgb = np.array(im).astype(np.uint8)
            bgr = rgb[:,:,[2,1,0]]
            video_writer.write(bgr)

        cv2.destroyAllWindows()
        video_writer.release()


class  Visualize_Obj():
    '''
    Util for visualizing the placement locations in the walkthrough, unshuffle, and end of unshuffle phases
    '''
    def __init__(self):
        pass

    def get_images(self, controller, target_objects, H, W, fov, label_colors):

        position_start = controller.last_event.metadata["agent"]["position"]
        rotation_start = controller.last_event.metadata["agent"]["rotation"]
        head_tilt = controller.last_event.metadata["agent"]["cameraHorizon"]

        event = controller.step(action="GetReachablePositions") 
        nav_pts = event.metadata["actionReturn"]
        nav_pts = np.array([list(d.values()) for d in nav_pts])

        image_dict = {}
        for i, obj in enumerate(target_objects):
            label_color = label_colors[i]
            obj_center = np.array(list(obj['axisAlignedBoundingBox']['center'].values()))

            # print(f"Getting image for {obj['objectId']}")

            dists = np.sqrt(np.sum((nav_pts - obj_center)**2, axis=1))
            argmin_pos = np.argmin(dists)
            closest_pos= nav_pts[argmin_pos] 

            # YAW calculation - rotate to object
            agent_to_obj = np.squeeze(obj_center) - (closest_pos + np.array([0.0, 0.675, 0.0]))
            agent_local_forward = np.array([0, 0, 1.0]) 
            flat_to_obj = np.array([agent_to_obj[0], 0.0, agent_to_obj[2]])
            flat_dist_to_obj = np.linalg.norm(flat_to_obj)
            flat_to_obj /= flat_dist_to_obj

            det = (flat_to_obj[0] * agent_local_forward[2]- agent_local_forward[0] * flat_to_obj[2])
            turn_angle = math.atan2(det, np.dot(agent_local_forward, flat_to_obj))

            # # add noise so not right in the center
            # noise = np.random.normal(0, 2, size=2)

            turn_yaw = np.degrees(turn_angle) #+ noise[0]

            turn_pitch = -np.degrees(math.atan2(agent_to_obj[1], flat_dist_to_obj)) #+ noise[1]

            event = controller.step('TeleportFull', position=dict(x=closest_pos[0], y=closest_pos[1], z=closest_pos[2]), rotation=dict(x=0.0, y=turn_yaw, z=0.0), horizon=turn_pitch, standing=True, forceAction=True)
            origin_T_camX = get_origin_T_camX(controller.last_event, False)

            pos_visit = []
            select = dists<=1.5
            dists2 = dists[select]
            nav_pts2 = nav_pts[select]
            
            start_index = 0
            has_seen_target = False
            
            rgbs = []
            while (not has_seen_target and start_index < len(dists2)) or len(nav_pts2) == 0:
                if len(nav_pts2)==0:
                    pos_visit += [closest_pos]
                else:
                    argmin_pos = np.argsort(dists2)[start_index]
                    closest_pos = nav_pts2[argmin_pos]
                    pos_visit += [closest_pos]
                # for p_i in range(len(pos_visit)):
                closest_pos = pos_visit[-1]
                # YAW calculation - rotate to object
                agent_to_obj = np.squeeze(obj_center) - (closest_pos + np.array([0.0, 0.675, 0.0]))
                agent_local_forward = np.array([0, 0, 1.0]) 
                flat_to_obj = np.array([agent_to_obj[0], 0.0, agent_to_obj[2]])
                flat_dist_to_obj = np.linalg.norm(flat_to_obj)
                flat_to_obj /= flat_dist_to_obj

                det = (flat_to_obj[0] * agent_local_forward[2]- agent_local_forward[0] * flat_to_obj[2])
                turn_angle = math.atan2(det, np.dot(agent_local_forward, flat_to_obj))

                # # add noise so not right in the center
                # noise = np.random.normal(0, 2, size=2)

                turn_yaw = np.degrees(turn_angle) #+ noise[0]

                turn_pitch = -np.degrees(math.atan2(agent_to_obj[1], flat_dist_to_obj)) #+ noise[1]

                controller.step('TeleportFull', position=dict(x=closest_pos[0], y=closest_pos[1]+0.675, z=closest_pos[2]), rotation=dict(x=0.0, y=turn_yaw, z=0.0), horizon=turn_pitch, standing=True, forceAction=True)
                origin_T_camX = get_origin_T_camX(controller.last_event, False)

                for obj_meta in controller.last_event.metadata["objects"]:
                    if obj_meta["name"] == obj['name'] and obj_meta["visible"]:
                        has_seen_target = True
                        # print('Has seen the target obj for visualization image')
                
                start_index += 1
                if len(nav_pts2) == 0:
                    break

            rgb = controller.last_event.frame

            hfov = float(fov) * np.pi / 180.
            pix_T_camX = np.array([
                [(W/2.)*1 / np.tan(hfov / 2.), 0., 0., 0.],
                [0., (H/2.)*1 / np.tan(hfov / 2.), 0., 0.],
                [0., 0.,  1, 0],
                [0., 0., 0, 1]])
            pix_T_camX[0,2] = W/2.
            pix_T_camX[1,2] = H/2.

            obj_3dbox_origin = get_3dbox_in_geom_format(obj)
            # get amodal box
            # origin_T_camX = get_origin_T_camX(controller.last_event, False)
            boxlist2d_amodal, obj_3dbox_camX = get_amodal2d(origin_T_camX.cuda(), obj_3dbox_origin.cuda(), torch.from_numpy(pix_T_camX).unsqueeze(0).cuda(), H, W)
            boxlist2d_amodal = boxlist2d_amodal.cpu().numpy()
            boxlist2d_amodal[[0,1]] = boxlist2d_amodal[[0,1]] - 5
            boxlist2d_amodal[[2,3]] = boxlist2d_amodal[[2,3]] + 5

            rect_th = 1
            img = rgb.copy()
            # cv2.rectangle(img, (int(boxlist2d_amodal[0]), int(boxlist2d_amodal[1])), (int(boxlist2d_amodal[2]), int(boxlist2d_amodal[3])), label_color, rect_th)

            img2 = np.zeros((img.shape[0]+5*2, img.shape[1]+5*2, 3)).astype(int)
            for i_i in range(3):
                img2[:,:,i_i] = np.pad(img[:,:,i_i], pad_width=5, constant_values=255)
            rgbs.append(img2)

            img = np.concatenate(rgbs, axis=1)

            objId = obj['name']
            if obj['parentReceptacles'] is None:
                receptacle = 'Floor'
            else:
                receptacle = obj['parentReceptacles'][-1]
            image_dict[objId] = {}
            image_dict[objId]['rgb'] = img
            image_dict[objId]['receptacle'] = receptacle.split('|')[0]

        controller.step('TeleportFull', position=dict(x=position_start["x"], y=position_start["y"], z=position_start["z"]), rotation=dict(x=rotation_start["x"], y=rotation_start["y"], z=rotation_start["z"]), horizon=head_tilt, standing=True, forceAction=True)

        return image_dict

    def get_walkthrough_images(self, walkthrough_explorer, controller):
        curr_objs_id_to_pose = walkthrough_explorer.get_current_objs_id_to_pose(stage = 'walkthrough')
        ordered_obj_ids = list(curr_objs_id_to_pose.keys())

        objs = controller.last_event.metadata["objects"]
        objs_dict = {}
        for obj in objs:
            objs_dict[obj["name"]] = obj

        target_objs = []
        label_colors = []
        for k in ordered_obj_ids:
            if k not in objs_dict.keys():
                continue
            target_objs.append(objs_dict[k])
            label_colors.append((0, 255, 0))
        
        self.walkthrough_images = self.get_images(controller, target_objs, args.H, args.W, args.fov, label_colors)

    def get_unshuffle_images(self, unshuffle_explorer, controller):
        # unshuffle_start_poses, walkthrough_start_poses, current_poses =  unshuffle_task.env.poses
        curr_objs_id_to_pose = unshuffle_explorer.get_current_objs_id_to_pose(stage = 'unshuffle')

        curr_energies_dict = unshuffle_explorer.get_curr_pose_difference_energy(stage='unshuffle')
        curr_energies = np.array(list(curr_energies_dict.values()))
        curr_misplaceds = curr_energies > 0.0
        where_misplaces = np.where(curr_misplaceds)[0]
        where_misplaces_ids = np.array(list(curr_energies_dict.keys()))[where_misplaces]

        ordered_obj_ids = []
        for o_i in list(where_misplaces_ids):
            unshuffle_obj = curr_objs_id_to_pose[o_i]
            ordered_obj_ids.append(unshuffle_obj["objectId"])

        objs = controller.last_event.metadata["objects"]
        objs_dict = {}
        for obj in objs:
            objs_dict[obj["objectId"]] = obj

        target_objs = []
        label_colors = []
        for k in ordered_obj_ids:
            if k not in objs_dict.keys():
                continue
            target_objs.append(objs_dict[k])
            label_colors.append((255, 0, 0))

        self.unshuffle_images = self.get_images(controller, target_objs, args.H, args.W, args.fov, label_colors)

    def get_rearranged_images(self, unshuffle_explorer, end_energy_dict):
        controller = unshuffle_explorer.controller

        curr_objs_id_to_pose = unshuffle_explorer.get_current_objs_id_to_pose(stage = 'rearrange')
        ordered_obj_ids = list(curr_objs_id_to_pose.keys())

        objs = controller.last_event.metadata["objects"]
        objs_dict = {}
        for obj in objs:
            objs_dict[obj["name"]] = obj

        target_objs = []
        label_colors = []
        for k in ordered_obj_ids:
            if k not in objs_dict.keys():
                continue
            target_objs.append(objs_dict[k])
            if end_energy_dict[k] > 0:
                label_colors.append((255, 0, 0))
            else:
                label_colors.append((0, 255, 0))
        
        self.rearrange_images = self.get_images(controller, target_objs, args.H, args.W, args.fov, label_colors = label_colors)

    def save_final_images(self, curr_task_id, tag):
        if not os.path.exists(args.movie_dir):
            os.mkdir(args.movie_dir)
        save_dir = os.path.join(args.movie_dir, curr_task_id)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        file_path = os.path.join(save_dir, f'{tag}.txt')
        with open(file_path, 'w') as f:
            pass
        obj_ids = list(self.unshuffle_images.keys())

        for obj_id in obj_ids:
            walkthrough_image = self.walkthrough_images[obj_id]['rgb']
            unshuffle_image = self.unshuffle_images[obj_id]['rgb']
            rearrange_image = self.rearrange_images[obj_id]['rgb']
            img = Image.fromarray(np.uint8(np.concatenate([walkthrough_image, unshuffle_image, rearrange_image], axis=1)))
            walkthrough_rec = self.walkthrough_images[obj_id]['receptacle']
            unshuffle_rec = self.unshuffle_images[obj_id]['receptacle']
            rearrange_rec = self.rearrange_images[obj_id]['receptacle']
            path = os.path.join(save_dir, f'{obj_id}_walkrec={walkthrough_rec}_unshuffrec={unshuffle_rec}_rearrec={rearrange_rec}.jpeg')
            print(f"saving {path}")
            img.save(path)

    def save_view_distance_vis(self, explorer, curr_task_id, stage):
        if not os.path.exists(args.movie_dir):
            os.mkdir(args.movie_dir)
        save_dir = os.path.join(args.movie_dir, curr_task_id)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        plt.figure('View_distance')
        m_vis = np.invert(explorer.mapper.get_traversible_map(explorer.selem_agent_radius, 1,loc_on_map_traversible=True))
        plt.imshow(m_vis * 0.8, origin='lower', vmin=0, vmax=1, cmap='Greys')
        map_for_distance = explorer.mapper.get_map_for_view_distance()
        norm = plt.Normalize(vmin = 0, vmax = 5)
        plt.imshow(map_for_distance, alpha = 0.7, origin='lower', interpolation='nearest', cmap=plt.get_cmap('jet'), norm = norm)
        ax=plt.gca()
        path = os.path.join(save_dir, f'{stage}_viewDist.jpeg')
        plt.savefig(path)
        print(f"saving {path}")
        plt.close('View_distance')
    
    def save_semantic_map(self, explorer, curr_task_id, stage):
        if not os.path.exists(args.movie_dir):
            os.mkdir(args.movie_dir)
        save_dir = os.path.join(args.movie_dir, curr_task_id)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        
        plt.figure('SemMap')

        selem =skimage.morphology.disk(int(2 * (0.05 / args.map_resolution)))
        map_topdown_seg = explorer.mapper.get_topdown_semantic_map()
        visited = explorer.mapper.get_visited()
        map_topdown_seg_vis = visualize_topdownSemanticMap(map_topdown_seg, explored, obstacle, visited)
        plt.imshow(map_topdown_seg_vis, origin = 'lower')
        path = os.path.join(save_dir, f'{stage}_semanticMap.jpeg')
        plt.savefig(path)
        print(f"saving {path}")
        plt.close('SemMap')

    def save_semantic_map(self, explorer, curr_task_id, stage):
        if not os.path.exists(args.movie_dir):
            os.mkdir(args.movie_dir)
        save_dir = os.path.join(args.movie_dir, curr_task_id)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        
        plt.figure('SemMap')

        selem =skimage.morphology.disk(int(2 * (0.05 / args.map_resolution)))
        map_topdown_seg = explorer.mapper.get_topdown_semantic_map()
        visited = explorer.mapper.get_visited()
        map_topdown_seg_vis = visualize_topdownSemanticMap(map_topdown_seg, explored, obstacle, visited)
        plt.imshow(map_topdown_seg_vis, origin = 'lower')
        path = os.path.join(save_dir, f'{stage}_semanticMap.jpeg')
        plt.savefig(path)
        print(f"saving {path}")
        plt.close('SemMap')

def visualize_segmentationRGB(rgb, segmented_dict = None, visualize_sem_seg = True, bbox = None):
    from detectron2.utils.visualizer import ColorMode, Visualizer
    from detectron2.structures import Boxes, Instances

    visualizer = Visualizer(rgb, instance_mode=ColorMode.IMAGE)
    H, W = rgb.shape[:2]
    if visualize_sem_seg:
        score_list = []
        category_list = [] 
        mask_list = []
        v_outputs = Instances(
        image_size=(H, W),
        scores=torch.tensor(segmented_dict['scores']),
        pred_classes=np.array(segmented_dict['categories']),
        pred_masks=torch.tensor(segmented_dict['masks']),
        )
        vis_output = visualizer.draw_instance_predictions(v_outputs.to("cpu"))
        vis_output = visualizer.draw_instance_predictions(v_outputs.to("cpu"))
        
    else:
        instances = Instances((H, W)).to(torch.device("cpu"))
        boxes = Boxes(bbox.to(torch.device("cpu")))
        instances.set('pred_boxes', boxes)
        vis_output = visualizer.draw_instance_predictions(predictions=instances)
    
    return vis_output.get_image()

def visualize_topdownSemanticMap(topdown_seg_map, explored, obstacle, visited):
    shape = topdown_seg_map.shape  #[H, W]
    color_palette = OTHER_PALETTE + CATEGORY_PALETTE
    topdown_seg_map = skimage.morphology.dilation(topdown_seg_map, skimage.morphology.disk(1))
    no_category_mask = topdown_seg_map == 0
    # add the fourth channel (0: out-of-bounds or no category, 1: Floor or explored , 2: Wall (Obstacle without category), 3: visited)
    topdown_seg_map[no_category_mask] = 0
    mask = np.logical_and(no_category_mask, explored)
    topdown_seg_map[mask] = 1
    obstacle = skimage.morphology.binary_dilation(obstacle,  skimage.morphology.disk(1)) == True    
    mask = np.logical_and(no_category_mask, obstacle)
    topdown_seg_map[mask] = 2
    # visited = skimage.morphology.dilation(visited, skimage.morphology.disk(2))
    topdown_seg_map[visited] = 3

    color_palette = [int(x * 255.) for x in color_palette]
    color_palette = np.uint8(color_palette).tolist()
    semantic_map = Image.new("P", (shape[1],shape[0]))
    semantic_map.putpalette(color_palette)
    semantic_map.putdata((topdown_seg_map.flatten()).astype(np.uint8))
    semantic_map = semantic_map.convert("RGBA")

    return semantic_map

def visualize_hierarchicalSemanticMap(hierarchical_seg_map, category_id, ax):
    category = CATEGORY_LIST[category_id-1]
    curr_map_instancesNum = np.max(hierarchical_seg_map)

    category_color = SEGMENTATION_CATEGORY_to_COLOR[category]
    category_color = RGB_to_Hex(category_color)
    ax.imshow(hierarchical_seg_map, cmap=colormap, origin='lower')





class ThorPositionTo2DFrameTranslator(object):
    def __init__(self, frame_shape, cam_position, orth_size):
        self.frame_shape = frame_shape
        self.lower_left = np.array((cam_position[0], cam_position[2])) - orth_size
        self.span = 2 * orth_size

    def __call__(self, position):
        if len(position) == 3:
            x, _, z = position
        else:
            x, z = position

        camera_position = (np.array((x, z)) - self.lower_left) / self.span
        return np.array(
            (
                round(self.frame_shape[0] * (1.0 - camera_position[1])),
                round(self.frame_shape[1] * camera_position[0]),
            ),
            dtype=int,
        )

def position_to_tuple(position):
    return (position["x"], position["y"], position["z"])

def add_agent_view_triangle(position, rotation, frame, pos_translator, trajectory_points=None, scale=1.0, opacity=0.7,color=None):
    p0 = np.array((position[0], position[2]))
    p1 = copy.copy(p0)
    p2 = copy.copy(p0)

    theta = -2 * math.pi * (rotation / 360.0)
    rotation_mat = np.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
    )
    offset1 = scale * np.array([-1, 1]) * math.sqrt(2) / 2
    offset2 = scale * np.array([1, 1]) * math.sqrt(2) / 2

    p1 += np.matmul(rotation_mat, offset1)
    p2 += np.matmul(rotation_mat, offset2)

    img1 = Image.fromarray(frame.astype("uint8"), "RGB").convert("RGBA")
    img2 = Image.new("RGBA", frame.shape[:-1])  # Use RGBA

    opacity = int(round(255 * opacity))  # Define transparency for the triangle.
    points = [tuple(reversed(pos_translator(p))) for p in [p0, p1, p2]]
    draw = ImageDraw.Draw(img2)
    draw.polygon(points, fill=(255, 255, 255, opacity))
    # trajectory_points.append(p0)
    # if len(trajectory_points) > 1:
    #      trace_points = [tuple(reversed(pos_translator(p))) for p in trajectory_points]
    #      draw.line(trace_points, color, width=2)

    img = Image.alpha_composite(img1, img2)
    return np.array(img.convert("RGB")), trajectory_points

def get_agent_map_data(c):
    c.step({"action": "ToggleMapView"})
    cam_postion = c.last_event.metadata["cameraPosition"]
    cam_orth_size = c.last_event.metadata["cameraOrthSize"]
    pos_translator = ThorPositionTo2DFrameTranslator(c.last_event.frame.shape, position_to_tuple(cam_postion), cam_orth_size)
    to_return = {
        "frame" : c.last_event.frame,
        "cam_position": cam_postion,
        "cam_orth_size": cam_orth_size,
        "pos_translator": pos_translator,
    }
    c.step({"action": "ToggleMapView"})
    return to_return

    


def drawTopImg(controller):
    # global location_points
    t = get_agent_map_data(controller)
    color = (0, 140, 255)
    new_frame, location_points = add_agent_view_triangle(
        position_to_tuple(controller.last_event.metadata["agent"]["position"]),
        controller.last_event.metadata["agent"]["rotation"]["y"],
        t["frame"],
        t["pos_translator"],
        trajectory_points=None,
        # location_points=None,
        color=color
    )
    return new_frame





