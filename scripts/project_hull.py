import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import json
import os
import pickle

def get_transform(vec1, vec2):
    T1 = np.zeros((4,4))
    T2 = np.zeros((4,4))
    T1[:3,:3] = R.from_rotvec(vec1[3:]).as_matrix()
    T2[:3,:3] = R.from_rotvec(vec2[3:]).as_matrix()
    T1[3, 3] = 1
    T2[3, 3] = 1
    T1[:3, 3] = vec1[:3]
    T2[:3, 3] = vec2[:3]
    transform = T2 @ np.linalg.inv(T1)

    return transform

def project_points_3d_to_2d(points_3d, intrinsic_matrix, extrinsic_matrix):
    """
    Projects 3D points to 2D using camera intrinsic and extrinsic matrices.
    """
    # Convert points to homogeneous coordinates
    points_3d_homogeneous = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])

    # Project points
    points_2d_homogeneous = intrinsic_matrix @ (extrinsic_matrix @ points_3d_homogeneous.T)[:3]
    
    # Convert back to non-homogeneous coordinates
    points_2d = points_2d_homogeneous[:2, :] / points_2d_homogeneous[2, :]
    return points_2d.T

def get_convex_hull_mask(points_2d, image_shape):
    """
    Computes the convex hull of the 2D points and creates a mask.
    """
    # Calculate convex hull
    hull = cv2.convexHull(points_2d.astype(np.int32))

    # Create mask
    mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, (255,255,255),lineType=cv2.LINE_AA)
    return mask

def get_dialated_mask(mask, kernel_size):
    """
    Dialates the mask.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(mask, kernel, iterations=1)

def get_negative_prompt(mask):
    mask = mask.astype(np.uint8)
    edge = cv2.Canny(mask, 100, 200)
    nx, ny = np.where(edge > 0)
    points = np.concatenate((nx[:, None], ny[:, None]), axis=1)

    return points


if __name__=='__main__':
    image_shape=[480,640]
    pts_path = ''
    extrinsic_path = ''
    save_path1 = ''
    save_path2 = ''
    save_path3 = ''
    prompt_save_path = ''
    image_path = ''
    vec1 = np.array([])
    vec2 = np.array([])
    intrinsic_matrix = np.array([])
    pts = np.loadtxt(pts_path)
    pts = pts[:,:3]
    transform = get_transform(vec1, vec2)
    with open(extrinsic_path, 'r') as file:
        data = json.load(file)
    center1 = []
    center2 = []
    negative = {}
    for i in range(len(data['frames'])):
        image_name = os.path.join(image_path, data['frames'][i]['file_path'])
        # image = cv2.imread(image_name)
        extrinsic_matrix = np.array(data['frames'][i]['transform_matrix'])
        extrinsic_matrix = np.linalg.inv(extrinsic_matrix)
        finetune_mask1 = np.zeros((480, 640))
        finetune_mask2 = np.zeros((480, 640))
        finetune_mask3 = np.zeros((480, 640))
        points_2d = project_points_3d_to_2d(pts, intrinsic_matrix, extrinsic_matrix) #N*3
        points_2d = points_2d.astype(np.int32)
        mask = get_convex_hull_mask(points_2d, image_shape)

        mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        image_ori = cv2.imread(image_name.replace('after0302', 'pcl_merge_data0302'))
        heat_map = cv2.addWeighted(image_ori, 0.8, mask_vis, 0.2, 0)
        save_name = 'vis0302/mask_before_' + str("%04d"%(i+1)) + '.png'
        cv2.imwrite(save_name, heat_map)

        x, y = np.where(mask > 0)
        center1.append([0.5 * (x.max() + x.min()), 0.5 * (y.max() + y.min())])
        finetune_mask1[mask > 0] = 1
        finetune_mask3[mask > 0] = 1
        mask_3d_pts = np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=1) @ transform[:3,:].T
        points_2d = project_points_3d_to_2d(mask_3d_pts, intrinsic_matrix, extrinsic_matrix) #N*3
        points_2d = points_2d.astype(np.int32)
        mask = get_convex_hull_mask(points_2d, image_shape)

        mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        image_ori = cv2.imread(image_name)
        heat_map = cv2.addWeighted(image_ori, 0.8, mask_vis, 0.2, 0)
        save_name = 'vis0302/mask_after_' + str("%04d"%(i+1)) + '.png'
        cv2.imwrite(save_name, heat_map)

        finetune_mask2[mask > 0] = 1
        finetune_mask3[mask > 0] = 1
        finetune_mask1_save = os.path.join(save_path1, data['frames'][i]['file_path'].split('/')[-1].replace('.png', '.npy'))
        finetune_mask2_save = os.path.join(save_path2, data['frames'][i]['file_path'].split('/')[-1].replace('.png', '.npy'))
        finetune_mask3_save = os.path.join(save_path3, data['frames'][i]['file_path'].split('/')[-1].replace('.png', '.npy'))
        np.save(finetune_mask3_save, finetune_mask3)

    print('end')