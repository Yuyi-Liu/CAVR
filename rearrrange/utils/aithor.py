import numpy as np
from rearrange_on_proc.utils import geom
import torch

def get_origin_T_camX(event, invert_pitch, standing=True, add_camera_height=True):
    if isinstance(event, dict):
        position = np.array(list(event["position"].values())) + np.array([0.0, 0.675, 0.0])
        rotation = np.array(list(event["rotation"].values()))
        rx = np.radians(event["horizon"]) # pitch
    else:
        if add_camera_height:
            position = np.array(list(event.metadata["agent"]["position"].values())) + np.array([0.0, 0.675, 0.0]) # adjust for camera height from agent
        else:
            position = np.array(list(event.metadata['cameraPosition'].values()))
        rotation = np.array(list(event.metadata["agent"]["rotation"].values()))
        rx = np.radians(event.metadata["agent"]["cameraHorizon"]) # pitch
    if invert_pitch: # in aithor negative pitch is up - turn this on if need the reverse
       rx = -rx 
    ry = np.radians(rotation[1]) # yaw
    rz = 0. # roll is always 0
    rotm = geom.eul2rotm_py(np.array([rx]), np.array([ry]), np.array([rz]))
    origin_T_camX = np.eye(4)
    origin_T_camX[0:3,0:3] = rotm
    origin_T_camX[0:3,3] = position
    origin_T_camX = torch.from_numpy(origin_T_camX)
    # st()
    # geom.apply_4x4(origin_T_camX.float(), torch.tensor([0.,0.,0.]).unsqueeze(0).unsqueeze(0).float())
    return origin_T_camX

def get_3dbox_in_geom_format(obj_meta):
    if obj_meta['objectOrientedBoundingBox'] is not None:
        obj_3dbox_origin = np.array(obj_meta['objectOrientedBoundingBox']['cornerPoints'])
        obj_3dbox_origin = torch.from_numpy(obj_3dbox_origin).unsqueeze(0).unsqueeze(0)
    else:
        obj_3dbox_origin = np.array(obj_meta['axisAlignedBoundingBox']['cornerPoints'])

        ry = 0. #obj_rot['x']) 
        rx = 0. #np.radians(obj_rot['y'])
        rz = 0. #np.radians(obj_rot['z'])
        xc = np.mean(obj_3dbox_origin[:,0])
        yc = np.mean(obj_3dbox_origin[:,1])
        zc = np.mean(obj_3dbox_origin[:,2])
        lx = np.max(obj_3dbox_origin[:,0]) - np.min(obj_3dbox_origin[:,0])
        ly = np.max(obj_3dbox_origin[:,1]) - np.min(obj_3dbox_origin[:,1])
        lz = np.max(obj_3dbox_origin[:,2]) - np.min(obj_3dbox_origin[:,2])
        box_origin = np.array([xc, yc, zc, lx, ly, lz, rx, ry, rz])
        box_origin = torch.from_numpy(box_origin).unsqueeze(0).unsqueeze(0)
        obj_3dbox_origin = geom.transform_boxes_to_corners(box_origin.cuda().float()).cpu().double()

    return obj_3dbox_origin


def get_amodal2d(origin_T_camX, obj_3dbox_origin, pix_T_camX, H, W):
    camX_T_origin = geom.safe_inverse_single(origin_T_camX)
    obj_3dbox_camX = geom.apply_4x4_to_corners(camX_T_origin.unsqueeze(0), obj_3dbox_origin)
    boxlist2d_amodal = geom.get_boxlist2d_from_corners_aithor(pix_T_camX, obj_3dbox_camX, H, W)[0][0]
    return boxlist2d_amodal, obj_3dbox_camX