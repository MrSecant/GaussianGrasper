import numpy as np
from PIL import Image
import open3d as o3d
import os
from scipy.spatial.transform import Rotation as R
import json
import cv2
import torch
import torch.nn.functional as F

def depth_image_to_point_cloud(depth_image, color_image, fx, fy, cx, cy, camera2base, mask):

    height, width = depth_image.shape 
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    Z = depth_image
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    mask = mask * (depth_image > 0.001) * (depth_image < 1.2)
    mask = mask > 0
    point_cloud = np.dstack((X, Y, Z)) 
    point_cloud = point_cloud[mask] 
    color = color_image[mask]  
    pts = merge_point_clouds(point_cloud, color, camera2base)

    return pts

def merge_point_clouds(points, colors, trans):
    column = np.ones((points.shape[0], 1))
    Tp_Nx4 = np.hstack((points, column))
    Tp_4xN = np.transpose(Tp_Nx4)
    matrix_Nx4 = np.dot(trans, Tp_4xN).T
    matrix_3columns = matrix_Nx4[:, :3]

    z_mask = (matrix_3columns[:, 2] > -0.3) *  (matrix_3columns[:, 2] < -0.1)
    merge = np.concatenate((matrix_3columns[z_mask, :], colors[z_mask, :].astype(np.uint8)), axis=1)
    
    return merge

def coloricp(source, target):
    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(source[:,:3])  # x, y, z
    src.colors = o3d.utility.Vector3dVector(source[:,3:]/255)  # R, G, B
    
    tar = o3d.geometry.PointCloud()
    tar.points = o3d.utility.Vector3dVector(target[:,:3])  # x, y, z
    tar.colors = o3d.utility.Vector3dVector(target[:,3:]/255)  # R, G, B

    voxel_radius = [0.02, 0.01, 0.005]
    max_iter = [30, 20, 10]
    # max_iter = [200, 100, 50]
    current_transformation = np.identity(4)

    for scale in range(3):
        iter = max_iter[scale]
        radius = voxel_radius[scale]

        source_down = src.voxel_down_sample(radius)
        target_down = tar.voxel_down_sample(radius)

        source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

        result_icp = o3d.pipelines.registration.registration_colored_icp(
            source_down, target_down, radius, current_transformation,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                            relative_rmse=1e-6,
                                                            max_iteration=iter))
        current_transformation = result_icp.transformation

    return current_transformation

def get_pts_and_normal(depth_path, image_path, normal_path, extrinsic_file, mask_path):
    fx, fy = 385.86016845703125, 385.3817443847656  # Focal lengths in x and y
    cx, cy = 325.68145751953125, 243.561767578125  # Principal point (image center)
    intrinsic = np.array([[385.86016845703125, 0.0, 325.68145751953125],
                          [0.0, 385.3817443847656, 243.561767578125],
                          [0.0, 0.0, 1.0]])
    camera2hand = np.array([[0.70710678,  0.70710678,   0,  0.008],
                            [-0.70710678, 0.70710678,   0, -0.008],
                            [0,           0,            1,  0.026],
                            [0,           0,            0,  1    ]])
    depth_list = sorted(os.listdir(depth_path))
    image_list = sorted(os.listdir(image_path))
    mask_list = sorted(os.listdir(mask_path))
    f = open(extrinsic_file, 'r')
    pts = []
    camera2base_list = []
    for i in range(len(depth_list)):
        param = f.readline().split(', ')
        rot_vec = np.array([float(param[3]), float(param[4]), float(param[5])])
        trans = np.array([float(param[0]), float(param[1]), float(param[2])]).T
        rot_mat = R.from_rotvec(rot_vec).as_matrix()
        hand2base = np.zeros((4,4))
        hand2base[:3,:3] = rot_mat
        hand2base[:3,3] = trans
        hand2base[3,3] = 1
        camera2base = np.dot(hand2base, camera2hand)
        camera2base_list.append(camera2base)
    f.close()
    for i in range(0, len(depth_list)):
        depth = np.load(depth_path + depth_list[i]) / 1000
        image = np.array(Image.open(image_path + image_list[i]))
        mask = np.load(mask_path + mask_list[i])
        point_cloud = depth_image_to_point_cloud(depth, image, fx, fy, cx, cy, camera2base_list[i], mask)
        pts.append(point_cloud)
    for i in range(len(depth_list)):
        depth = np.load(depth_path + depth_list[i]) / 1000
        save_path = normal_path + depth_list[i]
        cal_normal(depth, camera2base_list[i], save_path)
        # least_square_normal(depth, intrinsic, camera2base_list[i], save_path)
    return pts, camera2base_list

def gen_transforms(camera2base_list, save_path):
    camera = dict()
    camera['fl_x'] = 385.86016845703125
    camera['fl_y'] = 385.3817443847656
    camera['cx'] = 325.68145751953125
    camera['cy'] = 243.561767578125
    camera['w'] = 640
    camera['h'] = 480
    camera['camera_model'] = "OPENCV"
    camera['k1'] = -0.055006977170705795
    camera['k2'] = 0.06818309426307678
    camera['p1'] = -0.0007415282307192683
    camera['p2'] = 0.0006959497695788741
    frames = []
    for i in range(len(camera2base_list)):
        transform_dict = {}
        transform_dict["file_path"] = "images/images_" + str("%04d"%(i+1)) + '.png'
        transform_list = []
        for j in range(4):
            transform_list.append(camera2base_list[i][j, :].tolist())
        transform_dict["transform_matrix"] = transform_list
        frames.append(transform_dict)
    camera['frames'] = frames
    camera_save = json.dumps(camera)
    with open(save_path, "w") as file:
        file.write(camera_save)

def gen_image_info(camera2base_list, save_path):
    f = open(save_path, 'w')
    for i in range(len(camera2base_list)):
        line_list = []
        camera2base = camera2base_list[i]
        line_list.append(str(i + 1))
        rvec = R.from_matrix(camera2base[:3,:3]).as_quat().tolist()
        rvec = [rvec[3], rvec[0], rvec[1], rvec[2]]
        for j in range(len(rvec)):
            line_list.append(str(rvec[j]))
        tvec = camera2base[:3,3].tolist()
        for j in range(len(tvec)):
            line_list.append(str(tvec[j]))
        line_list.append('1')
        line_list.append('images_' + str("%04d"%(i+1)) + '.png')
        line = ' '.join(line_list) + '\n' + '0 0 0' + '\n'
        f.write(line)
        
def gen_camera_info(save_path):
    f = open(save_path, 'w')
    line_list = []
    line_list.append('1')
    line_list.append('OPENCV')
    line_list.append('640')
    line_list.append('480')
    line_list.append('385.86016845703125')
    line_list.append('385.3817443847656')
    line_list.append('325.68145751953125')
    line_list.append('243.561767578125')
    line_list.append('-0.05539723485708237')
    line_list.append('0.06696220487356186')
    line_list.append('-0.0005387895507737994')
    line_list.append('0.0007650373736396432')
    line = ' '.join(line_list) + '\n'
    f.write(line)

def cal_normal(depth, trans, save_path):
    fx, fy = 385.86016845703125, 385.3817443847656  # Focal lengths in x and y
    depth[depth < 0.01] = 1e-5
    dz_dv, dz_du = np.gradient(depth)  # u, v mean the pixel coordinate in the image
    du_dx = fx / depth  # x is xyz of camera coordinate
    dv_dy = fy / depth

    dz_dx = dz_du * du_dx
    dz_dy = dz_dv * dv_dy
    # cross-product (1,0,dz_dx)X(0,1,dz_dy) = (-dz_dx, -dz_dy, 1)
    normal_cross = np.dstack((-dz_dx, -dz_dy, np.ones_like(depth)))
    # normalize to unit vector
    normal_unit = normal_cross / np.linalg.norm(normal_cross, axis=2, keepdims=True)
    # set default normal to [0, 0, 1]

    normal_unit[~np.isfinite(normal_unit).all(2)] = [0, 0, 1]
    normal_unit = normal_unit.reshape(-1, 3).T
    normal_unit = np.dot(trans[:3,:3], normal_unit).T
    normal_unit = normal_unit.reshape((480, 640, 3))
    np.save(save_path, normal_unit)

    vis_normal = lambda normal: np.uint8((normal + 1) / 2 * 255)[..., ::-1]
    normal_vis = vis_normal(normal_unit)
    save_path_vis = save_path.replace('normals', 'normal_vis').replace('.npy', '.png')
    cv2.imwrite(save_path_vis, normal_vis) 

def get_points_coordinate(depth, instrinsic_inv, device="cuda"):
    B, height, width, C = depth.size()
    y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=device),
                           torch.arange(0, width, dtype=torch.float32, device=device)])
    y, x = y.contiguous(), x.contiguous()
    y, x = y.view(height * width), x.view(height * width)
    xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
    xyz = torch.unsqueeze(xyz, 0).repeat(B, 1, 1)  # [B, 3, H*W]
    xyz = torch.matmul(instrinsic_inv.float(), xyz) # [B, 3, H*W]
    depth_xyz = xyz * depth.view(B, 1, -1)  # [B, 3, Ndepth, H*W]

    return depth_xyz.view(B, 3, height, width)

def least_square_normal(depth, intrinsic, trans, save_path):
     # load depth & intrinsic
    H, W = depth.shape
    depth_torch = torch.from_numpy(depth).unsqueeze(0).unsqueeze(-1) # (B, h, w, 1)
    intrinsic_inv_np = np.linalg.inv(intrinsic)
    intrinsic_inv_torch = torch.from_numpy(intrinsic_inv_np).unsqueeze(0) # (B, 4, 4)

    ## step.2 compute matrix A
    # compute 3D points xyz
    points = get_points_coordinate(depth_torch, intrinsic_inv_torch[:, :3, :3], "cpu")
    point_matrix = F.unfold(points, kernel_size=5, stride=1, padding=4, dilation=2)

    # An = b
    matrix_a = point_matrix.view(1, 3, 25, H, W)  # (B, 3, 25, HxW)
    matrix_a = matrix_a.permute(0, 3, 4, 2, 1) # (B, HxW, 25, 3)
    matrix_a_trans = matrix_a.transpose(3, 4)
    matrix_b = torch.ones([1, H, W, 5, 1])

    # dot(A.T, A)
    point_multi = torch.matmul(matrix_a_trans, matrix_a)
    matrix_deter = torch.det(point_multi.to("cpu"))
    # make inversible
    inverse_condition = torch.ge(matrix_deter, 1e-5)
    inverse_condition = inverse_condition.unsqueeze(-1).unsqueeze(-1)
    inverse_condition_all = inverse_condition.repeat(1, 1, 1, 3, 3)
    # diag matrix to update uninverse
    diag_constant = torch.ones([3], dtype=torch.float32)
    diag_element = torch.diag(diag_constant)
    diag_element = diag_element.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    diag_matrix = diag_element.repeat(1, H, W, 1, 1)
    # inversible matrix
    inversible_matrix = torch.where(inverse_condition_all, point_multi, diag_matrix)
    inv_matrix = torch.inverse(inversible_matrix.to("cpu"))

    ## step.3 compute normal vector use least square
    # n = (A.T A)^-1 A.T b // || (A.T A)^-1 A.T b ||2
    generated_norm = torch.matmul(torch.matmul(inv_matrix, matrix_a_trans), matrix_b)
    norm_normalize = F.normalize(generated_norm, p=2, dim=3).numpy()
    normal_unit = norm_normalize.reshape(-1, 3).T
    normal_unit = np.dot(trans[:3,:3], normal_unit).T
    normal_unit = normal_unit.reshape((480, 640, 3))


    ## step.4 save normal vector
    # np.save(save_path, norm_normalize_np)
    norm_normalize_vis = (((normal_unit + 1) / 2) * 255)[..., ::-1].astype(np.uint8)
    save_path_vis = save_path.replace('normals', 'normal_vis').replace('.npy', '.png')
    cv2.imwrite(save_path_vis, norm_normalize_vis)

if __name__ == "__main__":
    path = ''
    depth_path = path + 'depth/'
    image_path = path + 'images/'
    normal_path = path + 'normals/'
    mask_path = path + 'boundary_mask/'
    extrinsic_file = path + 'hand2base.txt'
    transforms_save_path = path + 'transforms.json'
    camera_info_save_path = path + 'colmap/sparse/0/cameras.txt'
    images_info_save_path = path + 'colmap/sparse/0/images.txt'
    pts_save_path = path + 'colmap/sparse/0/points3D.txt'
    pts, camera2base_list = get_pts_and_normal(depth_path, image_path, normal_path, extrinsic_file, mask_path)
    gen_transforms(camera2base_list, transforms_save_path)
    gen_image_info(camera2base_list, images_info_save_path)
    gen_camera_info(camera_info_save_path)

    all_points = np.concatenate(pts)
    index = np.random.choice(np.arange(0, all_points.shape[0]), size=all_points.shape[0] // 8, replace=False)
    all_points = all_points[index]
    np.savetxt(path + 'all_points.txt', all_points)
    points_id = np.arange(1, all_points.shape[0] + 1).astype(np.int32)
    all_points = np.concatenate((points_id[:, None], all_points), axis=1)
    np.savetxt(pts_save_path, all_points)
