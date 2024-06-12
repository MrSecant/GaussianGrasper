# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python
"""
render.py
"""
from __future__ import annotations

import gzip
import json
import os
import re
import shutil
import struct
import sys
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import mediapy as media
import numpy as np
import torch
import tyro
from jaxtyping import Float
from rich import box, style
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from torch import Tensor
from typing_extensions import Annotated

from nerfstudio.cameras.camera_paths import (
    get_interpolated_camera_path,
    get_path_from_json,
    get_spiral_path,
)
from nerfstudio.cameras.cameras import Cameras, CameraType, RayBundle
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
    DataManagerConfig,
)
from nerfstudio.data.datasets.base_dataset import Dataset
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.data.utils.dataloaders import FixedIndicesEvalDataloader
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.model_components import renderers
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils import colormaps, install_checks
from nerfstudio.utils.eval_utils import eval_setup, eval_load_checkpoint
from nerfstudio.utils.rich_utils import CONSOLE, ItersPerSecColumn
from nerfstudio.utils.scripts import run_command
from nerfstudio.models.gaussian_splatting import GaussianSplattingModel
from gsplat._torch_impl import quat_to_rotmat
from torch.nn import Parameter
import yaml
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.transform import Rotation as R
from nerfstudio.utils.writer import EventName, TimeWriter
import tqdm
from nerfstudio.engine.optimizers import Optimizers
import copy
import shutil
import IPython

GS_KEYS = ["scales", "opacities", "colors_all", "means", "quats", "feature"]
GS_KEYS2 = ["xyz", "color", "opacity", "scaling", "rotation", "feature"]
vec1 = np.array([0,0,0,0,0,0]) # "x,y,z,rx,ry,rz" of end-effector in initial place
vec2 = np.array([0,0,0,0,0,0]) # "x,y,z,rx,ry,rz" of end-effector in targert place


@dataclass
class GaussianEditer:
    """Render all images in the dataset."""

    load_config: Path = Path(
        "/data/zyh/workspace/GS-Distilled-Feature-Fields/outputs/pcl_merge_data0301/gaussian-splatting/2024-02-29_205823/config.yml"
    )
    output_path: Optional[Path] = None  # Path("renders")
    """Path to output video file."""
    aabb= np.array([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]])
    edit_object_npy: Optional[Path] = None
    max_num_iterations: int = 580
    """how many iterations to finetune"""
    edit_result_dir = Path("nerfstudio_models_edit")
    edit_result_ckpt = Path("step-9999999999.ckpt")

    """x_min, y_min, z_min, x_max, y_max, z_max"""  # TODO make it to semantic selecting

    def __post_init__(self):
        if self.output_path is None:
            self.output_path = self.load_config.parent / "edit"

    def main(self):
        config: TrainerConfig
        # delete last edit checkpoint
        config = yaml.load(self.load_config.read_text(), Loader=yaml.Loader)
        last_edit_path = config.get_checkpoint_dir() / self.edit_result_ckpt
        last_edit_path.unlink(missing_ok=True)

        # create edit_result_dir
        edit_result_dir = config.get_base_dir() / self.edit_result_dir
        # edit_result_dir.unlink(missing_ok=True)
        shutil.rmtree(edit_result_dir, ignore_errors=True)
        edit_result_dir.mkdir(parents=True, exist_ok=True)
        config, pipeline, ori_checkpoint_path, _ = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=None,
            test_mode="inference",
        )
        transform_path = "/data/zyh/workspace/GS-Distilled-Feature-Fields/outputs/transform.json"
        with open(transform_path, "r") as file:
            transform = json.load(file)
        matrix = np.array(transform["transform_matrix"])
        scale = float(transform["scale"])

        data_manager_config = config.pipeline.datamanager
        assert isinstance(data_manager_config, DataManagerConfig)
        assert isinstance(pipeline.model, GaussianSplattingModel)
        model: GaussianSplattingModel = pipeline.model

        ## 1. transform gaussian
        if ".npy" in str(self.edit_object_npy):
            mask_3d_pts = np.load(str(self.edit_object_npy))[:, :3]
        elif ".txt" in str(self.edit_object_npy):
            mask_3d_pts = np.loadtxt(str(self.edit_object_npy))[:, :3]
        mask_3d_pts = np.concatenate((mask_3d_pts, np.ones((mask_3d_pts.shape[0], 1))), axis=1) @ matrix[:3, :].T
        mask_3d_pts *= scale
        mask3d = points_inside_convex_hull(model.means.detach(), mask_3d_pts)
        assert mask3d.sum() > 0, "no points in the box"
        # get transform
        transform = prepare_transform()
        transform = matrix @ transform @ np.linalg.inv(matrix)
        transform[:3, 3] *= scale
        transform = torch.from_numpy(transform).float()

        self.transformed_gs(model, mask3d, transform)

        # save start fintune gs
        self.save_checkpoint(edit_result_dir / "step-000000000.ckpt", ori_checkpoint_path, pipeline)

        # 2. finetune
        from nerfstudio.scripts.train import train_loop

        config_finetune = copy.copy(config)

        config_finetune.save_only_latest_checkpoint = False  # avoid overwriting
        config_finetune.max_num_iterations = self.max_num_iterations
        config_finetune.viewer.quit_on_train_completion = True
        config_finetune.relative_model_dir = self.edit_result_dir
        config_finetune.load_dir = edit_result_dir
        config_finetune.pipeline.datamanager.data = config_finetune.data.parent / "after_updating"
        config_finetune.pipeline.model.warmup_length = 300
        config_finetune.pipeline.model.densify_grad_thresh = 0.001
        config_finetune.pipeline.model.refine_every = 200

        train_loop(local_rank=0, world_size=0, config=config_finetune)

        # concat moved obj to the finetuned gs
        edit_checkpoint_path, _ = eval_load_checkpoint(config_finetune, pipeline)
        # self.concat_gs(pipeline.model, croped_gs)

        ## 3. save to file
        checkpoint_dir: Path = config.get_checkpoint_dir()
        CONSOLE.log(f"Saving checkpoints to: {checkpoint_dir}")
        self.save_checkpoint(checkpoint_dir / self.edit_result_ckpt, edit_checkpoint_path, pipeline)

    def cv2gl(self, xyz):
        pass

    @torch.no_grad()
    def get_croped_transformed_gs(self, model, mask3d, transform=torch.eye(4)):
        # Extracting subsets using the mask
        res = {}
        # TODO add parameter key here
        sub_keys = [k for k in GS_KEYS if k not in ["means", "quats"]]
        for k in sub_keys:
            res[k] = getattr(model, k)[mask3d]
        xyz_sub = model.means[mask3d]
        quats_sub = model.quats[mask3d]

        transform = transform.to(model.device)
        xyz_sub = torch.matmul(xyz_sub, transform[:3, :3].T) + transform[:3, 3]
        R = quat_to_rotmat(quats_sub)
        R = torch.matmul(transform[:3, :3], R)
        quats_sub = rotmat_to_quat(R)

        # # Construct nn.Parameters with specified gradients
        # model.means[mask3d] = xyz_sub
        # model.quats[mask3d] = quats_sub
        res["means"] = xyz_sub
        res["quats"] = quats_sub

        return res

    @torch.no_grad()
    def transformed_gs(self, model, mask3d, transform=torch.eye(4)):
        # Extracting subsets using the mask
        # res={}
        # TODO add parameter key here
        # sub_keys=[k for k in GS_KEYS if k not in ["means","quats"]]
        # for k in sub_keys:
        #     res[k]=getattr(model,k)[mask3d]
        xyz_sub = model.means[mask3d]
        quats_sub = model.quats[mask3d]

        transform = transform.to(model.device)
        xyz_sub = torch.matmul(xyz_sub, transform[:3, :3].T) + transform[:3, 3]
        R = quat_to_rotmat(quats_sub)
        R = torch.matmul(transform[:3, :3], R)
        quats_sub = rotmat_to_quat(R)

        # # Construct nn.Parameters with specified gradients
        means_data = model.means.data
        means_data[mask3d] = xyz_sub
        model.means = Parameter(means_data)

        quats_data = model.quats.data
        quats_data[mask3d] = quats_sub
        model.quats = Parameter(quats_data)

    def concat_gs(self, model, gs_dict):
        for k in GS_KEYS:
            setattr(model, k, Parameter(torch.cat([getattr(model, k), gs_dict[k]], dim=0)))

    def rm_gs(self, model, mask3d):
        for k in GS_KEYS:
            setattr(model, k, Parameter(getattr(model, k)[mask3d]))

    def save_checkpoint(self, ckpt_path, load_path, pipeline, optimizers=None):
        """Save the model and optimizers

        Args:
            step: number of steps in training for given checkpoint
        """
        # possibly make the checkpoint directory
        if not ckpt_path.parent.exists():
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        # save the checkpoint
        # ckpt_path: Path = checkpoint_dir / f"step-9999999999.ckpt"
        loaded_state = torch.load(load_path, map_location="cpu")
        loaded_state["step"] = 0
        loaded_state.update(
            {
                "pipeline": pipeline.state_dict(),
            }
        )
        if optimizers is not None:
            loaded_state.update(
                {
                    "optimizers": optimizers,
                }
            )

        torch.save(
            loaded_state,
            ckpt_path,
        )


def get_3d_mask(point_cloud, aabb):
    mask3d = np.zeros(point_cloud.shape[0], dtype=bool)
    mask3d = np.logical_or(mask3d, point_cloud[:, 0] > aabb[0, 0])
    mask3d = np.logical_and(mask3d, point_cloud[:, 0] < aabb[1, 0])
    mask3d = np.logical_and(mask3d, point_cloud[:, 1] > aabb[0, 1])
    mask3d = np.logical_and(mask3d, point_cloud[:, 1] < aabb[1, 1])
    mask3d = np.logical_and(mask3d, point_cloud[:, 2] > aabb[0, 2])
    mask3d = np.logical_and(mask3d, point_cloud[:, 2] < aabb[1, 2])
    assert mask3d.sum() > 0, "no points in the box"
    return torch.from_numpy(mask3d).to(point_cloud.device)


def points_inside_convex_hull(point_cloud, masked_points, remove_outliers=True, outlier_factor=1.0):
    """
    Given a point cloud and a mask indicating a subset of points, this function computes the convex hull of the
    subset of points and then identifies all points from the original point cloud that are inside this convex hull.

    Parameters:
    - point_cloud (torch.Tensor): A tensor of shape (N, 3) representing the point cloud.
    - mask (torch.Tensor): A tensor of shape (N,) indicating the subset of points to be used for constructing the convex hull.
    - remove_outliers (bool): Whether to remove outliers from the masked points before computing the convex hull. Default is True.
    - outlier_factor (float): The factor used to determine outliers based on the IQR method. Larger values will classify more points as outliers.

    Returns:
    - inside_hull_tensor_mask (torch.Tensor): A mask of shape (N,) with values set to True for the points inside the convex hull
                                            and False otherwise.
    """

    # Remove outliers if the option is selected
    if remove_outliers:
        Q1 = np.percentile(masked_points, 0, axis=0)
        Q3 = np.percentile(masked_points, 80, axis=0)
        IQR = Q3 - Q1
        outlier_mask = (masked_points < (Q1 - outlier_factor * IQR)) | (masked_points > (Q3 + outlier_factor * IQR))
        filtered_masked_points = masked_points[~np.any(outlier_mask, axis=1)]
    else:
        filtered_masked_points = masked_points

    # Compute the Delaunay triangulation of the filtered masked points
    delaunay = Delaunay(filtered_masked_points)

    # Determine which points from the original point cloud are inside the convex hull
    points_inside_hull_mask = delaunay.find_simplex(point_cloud.cpu().numpy()) >= 0

    # Convert the numpy mask back to a torch tensor and return
    inside_hull_tensor_mask = torch.tensor(points_inside_hull_mask, device="cuda")

    return inside_hull_tensor_mask


def rotmat_to_quat(rotmat: torch.Tensor) -> torch.Tensor:
    assert rotmat.shape[-2:] == (3, 3), rotmat.shape
    rotmat = rotmat.reshape(-1, 9)
    r00, r01, r02, r10, r11, r12, r20, r21, r22 = torch.unbind(rotmat, dim=(-1))
    w = torch.sqrt(1 + r00 + r11 + r22) / 2
    x = (r21 - r12) / (4 * w)
    y = (r02 - r20) / (4 * w)
    z = (r10 - r01) / (4 * w)
    return torch.stack([w, x, y, z], dim=-1)


def prepare_transform():
    ######### mock transform

    T1 = np.zeros((4, 4))
    T2 = np.zeros((4, 4))
    T1[:3, :3] = R.from_rotvec(vec1[3:]).as_matrix()
    T2[:3, :3] = R.from_rotvec(vec2[3:]).as_matrix()
    T1[3, 3] = 1
    T2[3, 3] = 1
    T1[:3, 3] = vec1[:3]
    T2[:3, 3] = vec2[:3]
    transform = T2 @ np.linalg.inv(T1)
    return transform


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(GaussianEditer).main()


if __name__ == "__main__":
    entrypoint()


def get_parser_fn():
    """Get the parser function for the sphinx docs."""
    return tyro.extras.get_parser(GaussianEditer)  # noqa
