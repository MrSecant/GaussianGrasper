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

"""
NeRF implementation that combines many recent advancements.
"""

from __future__ import annotations

import math
import pdb
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import viser.transforms as vtf
from pytorch_msssim import SSIM
from sklearn.neighbors import NearestNeighbors
from torch.nn import Parameter
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.camera_optimizers import (CameraOptimizer,
                                                  CameraOptimizerConfig)
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.callbacks import (TrainingCallback,
                                         TrainingCallbackAttributes,
                                         TrainingCallbackLocation)
from nerfstudio.engine.optimizers import Optimizers
from gsplat._torch_impl import quat_to_rotmat
from gsplat.nd_rasterize import NDRasterizeGaussians
from gsplat.project_gaussians import ProjectGaussians
from gsplat.rasterize import RasterizeGaussians
from gsplat.sh import SphericalHarmonics, num_sh_bases
# need following import for background color override
from nerfstudio.model_components import renderers
from nerfstudio.models.base_model import Model, ModelConfig


def random_quat_tensor(N):
    """
    Defines a random quaternion tensor of shape (N, 4)
    """
    u = torch.rand(N)
    v = torch.rand(N)
    w = torch.rand(N)
    return torch.stack(
        [
            torch.sqrt(1 - u) * torch.sin(2 * math.pi * v),
            torch.sqrt(1 - u) * torch.cos(2 * math.pi * v),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
        ],
        dim=-1,
    )

def RGB2SH(rgb):
    """
    Converts from RGB values [0,1] to the 0th spherical harmonic coefficient
    """
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    """
    Converts from the 0th spherical harmonic coefficient to RGB values [0,1]
    """
    C0 = 0.28209479177387814
    return sh * C0 + 0.5

def projection_matrix(znear, zfar, fovx, fovy, device:Union[str,torch.device]="cpu"):
    """
    Constructs an OpenGL-style perspective projection matrix.
    """
    t = znear * math.tan(0.5 * fovy)
    b = -t
    r = znear * math.tan(0.5 * fovx)
    l = -r
    n = znear
    f = zfar
    return torch.tensor(
        [
            [2 * n / (r - l), 0.0, (r + l) / (r - l), 0.0],
            [0.0, 2 * n / (t - b), (t + b) / (t - b), 0.0],
            [0.0, 0.0, (f + n) / (f - n), -1.0 * f * n / (f - n)],
            [0.0, 0.0, 1.0, 0.0],
        ],
        device=device,
    )

def norm_normalize(norm_out):
    norm_x, norm_y, norm_z = torch.split(norm_out, 1, dim=-1)
    norm = torch.sqrt(norm_x ** 2.0 + norm_y ** 2.0 + norm_z ** 2.0) + 1e-10
    final_out = torch.cat([norm_x / norm, norm_y / norm, norm_z / norm], dim=-1)
    return final_out

def cosine_similarity_loss(embeddings1, embeddings2):
    embeddings1 = F.normalize(embeddings1, dim=0)
    embeddings2 = F.normalize(embeddings2, dim=0)
    similarity = torch.sum(embeddings1 * embeddings2, dim=0)
    loss = 1 - similarity.mean()
    return loss

def sampling_in_mask(mask, sample_num):
    mask = mask.detach()
    nums = torch.unique(mask)
    index_list = []
    points_num = sample_num // (len(nums) - 1)
    for i in range(len(nums)):
        if nums[i] > -1:
            x, y = torch.where(mask == nums[i])
            num_points = min(points_num, x.shape[0])
            random_ids = torch.randperm(x.shape[0])[:num_points]
            pixels = torch.cat((x[random_ids].unsqueeze(0),y[random_ids].unsqueeze(0)), dim=0).permute(1,0)
            index_list.append(pixels)
    return torch.cat(index_list)

def sampling_pairs_in_mask(mask, sample_num):
    mask = mask.detach()
    nums = torch.unique(mask)
    selected_pairs = []
    for i in range(len(nums)):
        if nums[i] > -1:
            x, y = torch.where(mask == nums[i])
            num_pairs = min(sample_num, x.shape[0])
            indices1 = torch.randperm(x.shape[0])[:num_pairs]
            pixels_1 = torch.cat((x[indices1].unsqueeze(0),y[indices1].unsqueeze(0)), dim=0).permute(1,0)
            indices2 = torch.randperm(x.shape[0])[:num_pairs]
            pixels_2 = torch.cat((x[indices2].unsqueeze(0),y[indices2].unsqueeze(0)), dim=0).permute(1,0)
            selected_pairs.append([pixels_1, pixels_2])
            
    return selected_pairs

@dataclass
class GaussianSplattingModelConfig(ModelConfig):
    """Gaussian Splatting Model Config"""

    _target: Type = field(default_factory=lambda: GaussianSplattingModel)
    warmup_length: int = 500 ###change
    """period of steps where refinement is turned off"""
    refine_every: int = 100
    """period of steps where gaussians are culled and densified"""
    resolution_schedule: int = 250
    """training starts at 1/d resolution, every n steps this is doubled"""
    num_downscales: int = 1
    """at the beginning, resolution is 1/2^d, where d is this number"""
    cull_alpha_thresh: float = 0.1
    """threshold of opacity for culling gaussians"""
    cull_scale_thresh: float = 0.5
    """threshold of scale for culling gaussians"""
    reset_alpha_every: int = 30
    """Every this many refinement steps, reset the alpha"""
    densify_grad_thresh: float = 0.0002 ###change bigger
    """threshold of positional gradient norm for densifying gaussians"""
    densify_size_thresh: float = 0.01
    """below this size, gaussians are *duplicated*, otherwise split"""
    n_split_samples: int = 2
    """number of samples to split gaussians into"""
    sh_degree_interval: int = 1000
    """every n intervals turn on another sh degree"""
    cull_screen_size: float = 0.15
    """if a gaussian is more than this percent of screen space, cull it"""
    split_screen_size: float = 0.05
    """if a gaussian is more than this percent of screen space, split it"""
    stop_screen_size_at: int = 4000
    """stop culling/splitting at this step WRT screen size of gaussians"""
    random_init: bool = False
    """whether to initialize the positions uniformly randomly (not SFM points)"""
    ssim_lambda: float = 0.2
    """weight of ssim loss"""
    stop_split_at: int = 15000
    """stop splitting at this step"""
    sh_degree: int = 4
    """maximum degree of spherical harmonics to use"""
    camera_optimizer: CameraOptimizerConfig = CameraOptimizerConfig(mode="off")
    """camera optimizer config"""
    max_gauss_ratio: float = 10.0
    """threshold of ratio of gaussian max to min scale before applying regularization
    loss from the PhysGaussian paper
    """

class MLP(nn.Module):
    def __init__(self, in_dim=8, out_dim=512, hidden_list=[128]):
        super().__init__()
        layers = []
        lastv = in_dim
        if len(hidden_list) > 0:
            for hidden in hidden_list:
                layers.append(nn.Linear(lastv, hidden))
                layers.append(nn.ReLU())
                lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x

class Implicit_Field(nn.Module):
    def __init__(self, in_dim=3, out_dim=771, hidden_list=[128, 256, 512]):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x

class GaussianSplattingModel(Model):
    """Gaussian Splatting model

    Args:
        config: Gaussian Splatting configuration to instantiate model
    """

    config: GaussianSplattingModelConfig

    def __init__(self, *args, **kwargs):
        if "seed_points" in kwargs:
            self.seed_pts = kwargs["seed_points"]
        else:
            self.seed_pts = None
        super().__init__(*args, **kwargs)


    def populate_modules(self):
        if self.seed_pts is not None and not self.config.random_init:
            self.means = torch.nn.Parameter(self.seed_pts[0])  # (Location, Color)
        else:
            self.means = torch.nn.Parameter((torch.rand((500000, 3)) - 0.5) * 10)
        self.xys_grad_norm = None
        self.max_2Dsize = None
        self.normal_dim = 3
        self.clip_dim = 512
        self.feature_dim = 32 
        self.fea_up = MLP(self.feature_dim, self.clip_dim, hidden_list=[128])
        distances, _ = self.k_nearest_sklearn(self.means.data, 3)
        distances = torch.from_numpy(distances)
        # find the average of the three nearest neighbors for each point and use that as the scale
        avg_dist = distances.mean(dim=-1, keepdim=True)
        self.scales = torch.nn.Parameter(torch.log(avg_dist.repeat(1, 3)))
        self.quats = torch.nn.Parameter(random_quat_tensor(self.num_points))
        dim_sh = num_sh_bases(self.config.sh_degree)

        if self.seed_pts is not None and not self.config.random_init:
            fused_color = RGB2SH(self.seed_pts[1] / 255)
            shs = torch.zeros((fused_color.shape[0], dim_sh, 3)).float().cuda()
            shs[:, 0, :3] = fused_color
            shs[:, 1:, 3:] = 0.0
            self.colors_all = torch.nn.Parameter(shs)
        else:
            colors = torch.nn.Parameter(torch.rand(self.num_points, 1, 3))
            shs_rest = torch.nn.Parameter(torch.zeros((self.num_points, dim_sh - 1, 3)))
            self.colors_all = torch.nn.Parameter(torch.cat([colors, shs_rest], dim=1))

        self.opacities = torch.nn.Parameter(torch.logit(0.1 * torch.ones(self.num_points, 1)))
        feature_init = torch.rand(self.num_points, self.feature_dim) * 2 - 1
        self.feature = torch.nn.Parameter(feature_init)
        self.back_color = torch.zeros(self.feature_dim + 3)
        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        
        self.step = 0

        self.crop_box: Optional[OrientedBox] = None
        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )
    @property
    def colors(self):
        return SH2RGB(self.colors_all[:, 0, :])

    @property
    def shs_rest(self):
        return self.colors_all[:, 1:, :]

    def load_state_dict(self, dict, **kwargs): # type: ignore
        # resize the parameters to match the new number of points
        self.step = 30000
        newp = dict["means"].shape[0]
        self.means = torch.nn.Parameter(torch.zeros(newp, 3, device=self.device))
        self.scales = torch.nn.Parameter(torch.zeros(newp, 3, device=self.device))
        self.quats = torch.nn.Parameter(torch.zeros(newp, 4, device=self.device))
        self.opacities = torch.nn.Parameter(torch.zeros(newp, 1, device=self.device))
        self.colors_all = torch.nn.Parameter(
            torch.zeros(newp, num_sh_bases(self.config.sh_degree), 3, device=self.device)
        )
        self.feature = torch.nn.Parameter(torch.zeros(newp, self.feature_dim, device=self.device))
        super().load_state_dict(dict, **kwargs)

    def k_nearest_sklearn(self, x: torch.Tensor, k: int):
        """
        Find k-nearest neighbors using sklearn's NearestNeighbors.
        x: The data tensor of shape [num_samples, num_features]
        k: The number of neighbors to retrieve
        """
        # Convert tensor to numpy array
        x_np = x.cpu().numpy()

        # Build the nearest neighbors model
        nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean").fit(x_np)

        # Find the k-nearest neighbors
        distances, indices = nn_model.kneighbors(x_np)

        # Exclude the point itself from the result and return
        return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.float32)

    def remove_from_optim(self, optimizer, deleted_mask, new_params, group):
        """removes the deleted_mask from the optimizer provided"""
        assert len(new_params) == 1
        # assert isinstance(optimizer, torch.optim.Adam), "Only works with Adam"

        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        del optimizer.state[param]

        # Modify the state directly without deleting and reassigning.
        param_state["exp_avg"] = param_state["exp_avg"][~deleted_mask]
        param_state["exp_avg_sq"] = param_state["exp_avg_sq"][~deleted_mask]

        # Update the parameter in the optimizer's param group.
        del optimizer.param_groups[0]["params"][0]
        del optimizer.param_groups[0]["params"]
        optimizer.param_groups[0]["params"] = new_params
        optimizer.state[new_params[0]] = param_state

    def dup_in_optim(self, optimizer, dup_mask, new_params, n=2):
        """adds the parameters to the optimizer"""
        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        repeat_dims = (n,) + tuple(1 for _ in range(param_state["exp_avg"].dim() - 1))
        param_state["exp_avg"] = torch.cat(
            [param_state["exp_avg"], torch.zeros_like(param_state["exp_avg"][dup_mask.squeeze()]).repeat(*repeat_dims)],
            dim=0,
        )
        param_state["exp_avg_sq"] = torch.cat(
            [
                param_state["exp_avg_sq"],
                torch.zeros_like(param_state["exp_avg_sq"][dup_mask.squeeze()]).repeat(*repeat_dims),
            ],
            dim=0,
        )
        del optimizer.state[param]
        optimizer.state[new_params[0]] = param_state
        optimizer.param_groups[0]["params"] = new_params
        del param

    def after_train(self, step: int):
        with torch.no_grad():
            # keep track of a moving average of grad norms
            visible_mask = (self.radii > 0).flatten()
            grads = self.xys.grad.detach().norm(dim=-1)  # TODO fill in
            # print(f"grad norm min {grads.min().item()} max {grads.max().item()} mean {grads.mean().item()} size {grads.shape}")
            if self.xys_grad_norm is None:
                self.xys_grad_norm = grads
                self.vis_counts = torch.ones_like(self.xys_grad_norm)
            else:
                assert self.vis_counts is not None
                self.vis_counts[visible_mask] = self.vis_counts[visible_mask] + 1
                self.xys_grad_norm[visible_mask] = grads[visible_mask] + self.xys_grad_norm[visible_mask]

            # update the max screen size, as a ratio of number of pixels
            if self.max_2Dsize is None:
                self.max_2Dsize = torch.zeros_like(self.radii, dtype=torch.float32)
            newradii = self.radii.detach()[visible_mask]
            self.max_2Dsize[visible_mask] = torch.maximum(
                self.max_2Dsize[visible_mask], newradii / float(max(self.last_size[0], self.last_size[1]))
            )

    def set_crop(self, crop_box: Optional[OrientedBox]):
        self.crop_box = crop_box

    def set_background(self, back_color: torch.Tensor):
        assert back_color.shape == (3,)
        self.back_color = back_color

    def refinement_after(self, optimizers: Optimizers, step):
        if self.step >= self.config.warmup_length:
            with torch.no_grad():
                # only split/cull if we've seen every image since opacity reset
                reset_interval = self.config.reset_alpha_every * self.config.refine_every
                if (
                    self.step < self.config.stop_split_at
                    and self.step % reset_interval > self.num_train_data + self.config.refine_every
                ):
                    # then we densify
                    assert self.xys_grad_norm is not None and self.vis_counts is not None and self.max_2Dsize is not None
                    avg_grad_norm = (
                        (self.xys_grad_norm / self.vis_counts) * 0.5 * max(self.last_size[0], self.last_size[1])
                    )
                    high_grads = (avg_grad_norm > self.config.densify_grad_thresh).squeeze()
                    splits = (self.scales.exp().max(dim=-1).values > self.config.densify_size_thresh).squeeze()
                    if self.step < self.config.stop_screen_size_at:
                        splits |= (self.max_2Dsize > self.config.split_screen_size).squeeze()
                    splits &= high_grads
                    nsamps = self.config.n_split_samples
                    (
                        split_means,
                        split_colors,
                        split_opacities,
                        split_scales,
                        split_quats,
                        split_feature,
                    ) = self.split_gaussians(splits, nsamps)

                    dups = (self.scales.exp().max(dim=-1).values <= self.config.densify_size_thresh).squeeze()
                    dups &= high_grads
                    dup_means, dup_colors, dup_opacities, dup_scales, dup_quats, dup_feature = self.dup_gaussians(dups)
                    self.means = Parameter(torch.cat([self.means.detach(), split_means, dup_means], dim=0))
                    self.colors_all = Parameter(torch.cat([self.colors_all.detach(), split_colors, dup_colors], dim=0))
                    self.feature = Parameter(torch.cat([self.feature.detach(), split_feature, dup_feature], dim=0))
                    self.opacities = Parameter(torch.cat([self.opacities.detach(), split_opacities, dup_opacities], dim=0))
                    self.scales = Parameter(torch.cat([self.scales.detach(), split_scales, dup_scales], dim=0))
                    self.quats = Parameter(torch.cat([self.quats.detach(), split_quats, dup_quats], dim=0))
                    # append zeros to the max_2Dsize tensor
                    self.max_2Dsize = torch.cat([self.max_2Dsize, torch.zeros_like(split_scales[:, 0]), torch.zeros_like(dup_scales[:, 0])], dim=0)
                    split_idcs = torch.where(splits)[0]
                    param_groups = self.get_gaussian_param_groups()
                    for group, param in param_groups.items():
                        if 'net' not in group:
                            self.dup_in_optim(optimizers.optimizers[group], split_idcs, param, n=nsamps)
                    dup_idcs = torch.where(dups)[0]

                    param_groups = self.get_gaussian_param_groups()
                    for group, param in param_groups.items():
                        if 'net' not in group:
                            self.dup_in_optim(optimizers.optimizers[group], dup_idcs, param, 1)

                # Offset all the opacity reset logic by refine_every so that we don't
                # save checkpoints right when the opacity is reset (saves every 2k)
                if self.step % reset_interval > self.num_train_data + self.config.refine_every:
                # if False:
                    # then cull
                    deleted_mask = self.cull_gaussians()
                    param_groups = self.get_gaussian_param_groups()
                    for group, param in param_groups.items():
                        if 'net' not in group:
                            self.remove_from_optim(optimizers.optimizers[group], deleted_mask, param, group)

                if self.step % reset_interval == self.config.refine_every:
                    reset_value = self.config.cull_alpha_thresh * 0.8
                    self.opacities.data = torch.full_like(
                        self.opacities.data, torch.logit(torch.tensor(reset_value)).item()
                    )
                    # reset the exp of optimizer
                    optim = optimizers.optimizers["opacity"]
                    param = optim.param_groups[0]["params"][0]
                    param_state = optim.state[param]
                    param_state["exp_avg"] = torch.zeros_like(param_state["exp_avg"])
                    param_state["exp_avg_sq"] = torch.zeros_like(param_state["exp_avg_sq"])
                self.xys_grad_norm = None
                self.vis_counts = None
                self.max_2Dsize = None

    def cull_gaussians(self):
        """
        This function deletes gaussians with under a certain opacity threshold
        """
        n_bef = self.num_points
        # cull transparent ones
        culls = (torch.sigmoid(self.opacities) < self.config.cull_alpha_thresh).squeeze()
        if self.step > self.config.refine_every * self.config.reset_alpha_every:
            # cull huge ones
            toobigs = (torch.exp(self.scales).max(dim=-1).values > self.config.cull_scale_thresh).squeeze()
            culls = culls | toobigs
            if self.step < self.config.stop_screen_size_at:
                # cull big screen space
                assert self.max_2Dsize is not None
                culls = culls | (self.max_2Dsize > self.config.cull_screen_size).squeeze()
        self.means = Parameter(self.means[~culls].detach())
        self.scales = Parameter(self.scales[~culls].detach())
        self.quats = Parameter(self.quats[~culls].detach())
        self.colors_all = Parameter(self.colors_all[~culls].detach())
        self.opacities = Parameter(self.opacities[~culls].detach())
        self.feature = Parameter(self.feature[~culls].detach())
        print(f"Culled {n_bef - self.num_points} gaussians")
        return culls

    def split_gaussians(self, split_mask, samps):
        """
        This function splits gaussians that are too large
        """

        n_splits = split_mask.sum().item()
        print(f"Splitting {split_mask.sum().item()/self.num_points} gaussians: {n_splits}/{self.num_points}")
        centered_samples = torch.randn((samps * n_splits, 3), device=self.device)  # Nx3 of axis-aligned scales
        scaled_samples = (
            torch.exp(self.scales[split_mask].repeat(samps, 1)) * centered_samples
        )  # how these scales are rotated
        quats = self.quats[split_mask] / self.quats[split_mask].norm(dim=-1, keepdim=True)  # normalize them first
        rots = quat_to_rotmat(quats.repeat(samps, 1))  # how these scales are rotated
        rotated_samples = torch.bmm(rots, scaled_samples[..., None]).squeeze()
        new_means = rotated_samples + self.means[split_mask].repeat(samps, 1)
        # step 2, sample new colors
        new_colors_all = self.colors_all[split_mask].repeat(samps, 1, 1)
        new_feature = self.feature[split_mask].repeat(samps, 1)
        # step 3, sample new opacities
        new_opacities = self.opacities[split_mask].repeat(samps, 1)
        # step 4, sample new scales
        size_fac = 1.6
        new_scales = torch.log(torch.exp(self.scales[split_mask]) / size_fac).repeat(samps, 1)
        self.scales[split_mask] = torch.log(torch.exp(self.scales[split_mask]) / size_fac)
        # step 5, sample new quats
        new_quats = self.quats[split_mask].repeat(samps, 1)

        return new_means, new_colors_all, new_opacities, new_scales, new_quats, new_feature

    def dup_gaussians(self, dup_mask):
        """
        This function duplicates gaussians that are too small
        """
        n_dups = dup_mask.sum().item()
        print(f"Duplicating {dup_mask.sum().item()/self.num_points} gaussians: {n_dups}/{self.num_points}")
        dup_means = self.means[dup_mask]
        dup_colors = self.colors_all[dup_mask]
        dup_opacities = self.opacities[dup_mask]
        dup_scales = self.scales[dup_mask]
        dup_quats = self.quats[dup_mask]
        dup_feature = self.feature[dup_mask]

        return dup_means, dup_colors, dup_opacities, dup_scales, dup_quats, dup_feature

    @property
    def num_points(self):
        return self.means.shape[0]

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        cbs = []
        cbs.append(TrainingCallback([TrainingCallbackLocation.BEFORE_TRAIN_ITERATION], self.step_cb))
        # The order of these matters
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.after_train,
            )
        )
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.refinement_after,
                update_every_num_iters=self.config.refine_every,
                args=[training_callback_attributes.optimizers],
            )
        )
        return cbs

    def step_cb(self, step):
        self.step = step

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        return {
            "xyz": [self.means],
            "color": [self.colors_all],
            "opacity": [self.opacities],
            "scaling": [self.scales],
            "rotation": [self.quats],
            "feature": [self.feature],
            "up_net": [p for p in list(self.fea_up.parameters())],
        }

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """
        gps = self.get_gaussian_param_groups()
        # add camera optimizer param groups
        self.camera_optimizer.get_param_groups(gps)
        return gps

    def _get_downscale_factor(self):
        if self.training:
            return 2 ** max((self.config.num_downscales - self.step // self.config.resolution_schedule), 0)
        else:
            return 1

    def get_smallest_axis(self, return_idx=False):
        """Returns the smallest axis of the Gaussians.

        Args:
            return_idx (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        rotation_matrices = quat_to_rotmat(self.quats)
        smallest_axis_idx = self.scales.exp().min(dim=-1)[1][..., None, None].expand(-1, 3, -1)
        smallest_axis = rotation_matrices.gather(2, smallest_axis_idx)
        if return_idx:
            return smallest_axis.squeeze(dim=2), smallest_axis_idx[..., 0, 0]
        return smallest_axis.squeeze(dim=2)

    def get_normals(self):
        return self.get_smallest_axis()

    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}
        assert camera.shape[0] == 1, "Only one camera at a time"
        if self.training:
            # currently relies on the branch vickie/camera-grads
            self.camera_optimizer.apply_to_camera(camera)
        if self.training:
            background = torch.rand(self.feature_dim, device=self.device)
        else:
            # logic for setting the background of the scene
            if renderers.BACKGROUND_COLOR_OVERRIDE is not None:
                background = renderers.BACKGROUND_COLOR_OVERRIDE
            else:
                background = self.back_color.to(self.device)
        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                return {"rgb": background.repeat(camera.height.item(), camera.width.item(), 1)}
        else:
            crop_ids = None
        camera_downscale = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_downscale)
        # shift the camera to center of scene looking at center
        R = camera.camera_to_worlds[0, :3, :3]  # 3 x 3
        T = camera.camera_to_worlds[0, :3, 3:4]  # 3 x 1
        # flip the z axis to align with gsplat conventions
        R_edit = torch.tensor(vtf.SO3.from_x_radians(np.pi).as_matrix(), device=R.device, dtype=R.dtype)
        R = R @ R_edit
        # analytic matrix inverse to get world2camera matrix
        R_inv = R.T
        T_inv = -R_inv @ T
        viewmat = torch.eye(4, device=R.device, dtype=R.dtype)
        viewmat[:3, :3] = R_inv
        viewmat[:3, 3:4] = T_inv
        # calculate the FOV of the camera given fx and fy, width and height
        cx = camera.cx.item()
        cy = camera.cy.item()
        fovx = 2 * math.atan(camera.width / (2 * camera.fx))
        fovy = 2 * math.atan(camera.height / (2 * camera.fy))
        W, H = camera.width.item(), camera.height.item()
        self.last_size = (H, W)
        projmat = projection_matrix(0.001, 1000, fovx, fovy, device=self.device)
        BLOCK_X, BLOCK_Y = 16, 16
        tile_bounds = (
            (W + BLOCK_X - 1) // BLOCK_X,
            (H + BLOCK_Y - 1) // BLOCK_Y,
            1,
        )

        if crop_ids is not None:
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]
            colors_crop = self.colors_all[crop_ids]
            feature_crop = self.feature[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
        else:
            opacities_crop = self.opacities
            means_crop = self.means
            colors_crop = self.colors_all
            feature_crop = self.feature
            scales_crop = self.scales
            quats_crop = self.quats
        
        self.xys, depths, self.radii, conics, num_tiles_hit, cov3d = ProjectGaussians.apply(
            means_crop,
            torch.exp(scales_crop),
            1,
            quats_crop / quats_crop.norm(dim=-1, keepdim=True),
            viewmat.squeeze()[:3, :],
            projmat.squeeze() @ viewmat.squeeze(),
            camera.fx.item(),
            camera.fy.item(),
            cx,
            cy,
            H,
            W,
            tile_bounds,
        )
        if (self.radii).sum() == 0:
            return {"rgb": background.repeat(camera.height.item(), camera.width.item(), 1)}
        
        # mask = torch.ones(len(self.xys)).to(self.device)
        # mask = torch.logical_and(mask,self.xys[:,0] > 0)
        # mask = torch.logical_and(mask,self.xys[:,1] > 0)
        # mask = torch.logical_and(mask,self.xys[:,0] < W)
        # mask = torch.logical_and(mask,self.xys[:,1] < H)

        # Important to allow xys grads to populate properly
        if self.training:
            self.xys.retain_grad()
        if self.config.sh_degree > 0:
            viewdirs = means_crop.detach() - camera.camera_to_worlds.detach()[..., :3, 3]  # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            n = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
            rgbs = SphericalHarmonics.apply(n, viewdirs, colors_crop)
            rgbs = torch.clamp(rgbs + 0.5, 0.0, 1.0)
        else:
            rgbs = self.get_colors.squeeze()  # (N, 3)
            rgbs = torch.sigmoid(rgbs)
        rgb = RasterizeGaussians.apply(
            self.xys,
            depths,
            self.radii,
            conics,
            num_tiles_hit,
            rgbs,
            torch.sigmoid(opacities_crop),
            H,
            W,
            torch.zeros(3, device=self.device),
            )
        feature = NDRasterizeGaussians.apply(
            self.xys,
            depths,
            self.radii,
            conics,
            num_tiles_hit,
            feature_crop,
            torch.sigmoid(opacities_crop),
            H,
            W,
            torch.zeros(self.feature_dim, device=self.device),
        )
        depth_im = RasterizeGaussians.apply(
            self.xys,
            depths,
            self.radii,
            conics,
            num_tiles_hit,
            depths[:, None].repeat(1, 3),
            torch.sigmoid(opacities_crop),
            H,
            W,
            torch.ones(3, device=self.device) * 10,
            )[..., 0:1]
        
        self.normals= self.get_normals()
        normal_im=RasterizeGaussians.apply(
            self.xys,
            depths,
            self.radii,
            conics,
            num_tiles_hit,
            self.normals,
            torch.sigmoid(opacities_crop),
            H,
            W,
            torch.zeros(3, device=self.device),
        )
        normal_vis = (normal_im + 1) / 2
        
        # vis feature
        if feature_crop.shape[1] == 3:
            feature_vis = F.normalize(feature, dim=-1)
            feature_vis = (feature_vis + 1) / 2
        else:
            feature_vis = feature.view(-1, feature.size(-1))
            U, S, V = torch.pca_lowrank(feature_vis, q=3)
            transformed_tensor = torch.matmul(feature_vis, V[:, :3])
            feature_vis = transformed_tensor.view(feature.size()[:-1] + (3,))
        
        # rescale the camera back to original dimensions
        camera.rescale_output_resolution(camera_downscale)
        
        
        return {"rgb":rgb, "feature": feature, "depth": depth_im, "normal":normal_im, 
                "normal_vis": normal_vis, "feature_vis": feature_vis}

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        d = self._get_downscale_factor()
        if d > 1:
            newsize = [batch["image"].shape[0] // d, batch["image"].shape[1] // d]
            gt_img = TF.resize(batch["image"].permute(2, 0, 1), newsize, antialias=None).permute(1, 2, 0)
        else:
            gt_img = batch["image"]
        metrics_dict = {}
        gt_rgb = gt_img.to(self.device)  # RGB or RGBA image
        gt_normal = batch["normal"].permute(2, 0, 1).unsqueeze(0).to(self.device)
        gt_normal = F.interpolate(gt_normal, size=(gt_img.shape[0], gt_img.shape[1]), mode='bilinear').squeeze(0)
        gt_depth = batch["depth"].permute(2, 0, 1).unsqueeze(0).to(self.device)
        depth_mask = (gt_depth > 0.01) * 1.0 
        gt_depth = F.interpolate(gt_depth, size=(gt_img.shape[0], gt_img.shape[1]), mode='bilinear').squeeze(0)
        depth_mask = F.interpolate(depth_mask, size=(gt_img.shape[0], gt_img.shape[1]), mode='nearest').squeeze(0)
        gt_mask = batch["sam_mask"].to(self.device)
        gt_mask = F.interpolate(gt_mask.float().unsqueeze(0).unsqueeze(0), size=(gt_img.shape[0], gt_img.shape[1]), 
                                mode='nearest').squeeze(0).squeeze(0)
        valid_mask = batch["valid_mask"].to(self.device)
        valid_mask = F.interpolate(valid_mask.float().unsqueeze(0).unsqueeze(0), size=(gt_img.shape[0], gt_img.shape[1]), 
                                mode='nearest').squeeze(0).squeeze(0)
        depth_mask = depth_mask * valid_mask
        depth_mask = depth_mask > 0
        valid_mask = valid_mask > 0
        gt_mask[~valid_mask] = -1.0
        metrics_dict["psnr"] = self.psnr(outputs["rgb"][valid_mask, :], gt_rgb[valid_mask, :])
        self.camera_optimizer.get_metrics_dict(metrics_dict)
        metrics_dict["gaussian_count"] = self.num_points

        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        d = self._get_downscale_factor()
        if d > 1:
            newsize = [batch["image"].shape[0] // d, batch["image"].shape[1] // d]
            gt_img = TF.resize(batch["image"].permute(2, 0, 1), newsize, antialias=None).permute(1, 2, 0)
        else:
            gt_img = batch["image"]
        gt_normal = batch["normal"].permute(2, 0, 1).unsqueeze(0).to(self.device)
        gt_normal = F.interpolate(gt_normal, size=(gt_img.shape[0], gt_img.shape[1]), mode='bilinear').squeeze(0)
        gt_normal = F.normalize(gt_normal, dim=0)
        gt_depth = batch["depth"].permute(2, 0, 1).unsqueeze(0).to(self.device)
        depth_mask = (gt_depth > 0.05) * 1.0
        gt_depth = F.interpolate(gt_depth, size=(gt_img.shape[0], gt_img.shape[1]), mode='bilinear').squeeze(0)
        depth_mask = F.interpolate(depth_mask, size=(gt_img.shape[0], gt_img.shape[1]), mode='nearest').squeeze(0)
        gt_mask = batch["sam_mask"].to(self.device)
        gt_mask = F.interpolate(gt_mask.float().unsqueeze(0).unsqueeze(0), size=(gt_img.shape[0], gt_img.shape[1]), 
                                mode='nearest').squeeze(0).squeeze(0)
        valid_mask = batch["valid_mask"].to(self.device)
        valid_mask = F.interpolate(valid_mask.float().unsqueeze(0).unsqueeze(0), size=(gt_img.shape[0], gt_img.shape[1]), 
                                mode='nearest').squeeze(0).squeeze(0)
        depth_mask = depth_mask * valid_mask
        depth_mask = depth_mask > 0
        valid_mask = valid_mask > 0
        gt_mask[~valid_mask] = -1.0
        # sam_valid_mask = gt_mask > -1.0
        # selected_points = sampling_in_mask(gt_mask)
        gt_fea = batch["feature"].permute(2, 0, 1).float().to(self.device)
        gt_fea = F.interpolate(gt_fea.unsqueeze(0), size=(gt_img.shape[0], gt_img.shape[1]), mode='nearest').squeeze(0)
        
        normal = outputs["normal"].permute(2,0,1)
        depth = outputs["depth"].permute(2, 0, 1)
        normal_loss = 0.5 * F.mse_loss(normal[:, depth_mask[0]], gt_normal[:, depth_mask[0]], reduction='mean') + 0.5 * cosine_similarity_loss(normal[:, depth_mask[0]], gt_normal[:, depth_mask[0]])
        depth_loss = F.l1_loss(depth[depth_mask], gt_depth[depth_mask], reduction='mean')
        
        Ll1 = torch.abs(gt_img[valid_mask, :] - outputs["rgb"][valid_mask, :]).mean()
        gt_img[~valid_mask, :] = 0.0
        outputs["rgb"][~valid_mask, :] = 0.0
        simloss = 1 - self.ssim(gt_img.permute(2, 0, 1)[None, ...], outputs["rgb"].permute(2, 0, 1)[None, ...])
        
        '''pixel level'''
        # fea_out = self.fea_up(outputs["feature"])
        # fea_out = outputs["feature"]
        # predicted_fea = fea_out.permute(2, 0, 1)
        # fea_loss = cosine_similarity_loss(predicted_fea[:, valid_mask], gt_fea[:, valid_mask])
        # fea_loss = cosine_similarity_loss(predicted_fea, gt_fea)
        # selected_points = sampling_in_mask(gt_mask, 1000)
        # feature = outputs["feature"]
        # fea_up = self.fea_up(feature[selected_points[:,0], selected_points[:,1], :]).permute(1,0)
        # fea_loss = cosine_similarity_loss(fea_up, gt_fea[:, selected_points[:,0] , selected_points[:,1]])

        '''low rank decomposition'''
        # feature = F.normalize(outputs["feature"], dim=-1)
        # normal = F.normalize(outputs["normal"], dim=-1).permute(2,0,1)
        # fea_up, fea_down = self.fea_up(gt_fea.permute(1, 2, 0), selected_points)
        # up_loss = cosine_similarity_loss(fea_up, gt_fea[:, selected_points[:,0] , selected_points[:,1]]) \
                # + 0.5 * F.mse_loss(fea_up, gt_fea[:, selected_points[:,0] , selected_points[:,1]], reduction='mean')
        # fea_loss = cosine_similarity_loss(outputs["feature"].permute(2, 0, 1)[:, sam_valid_mask], fea_down[:, sam_valid_mask]) \
                #  + 0.5 * F.mse_loss(outputs["feature"].permute(2, 0, 1)[:, sam_valid_mask], fea_down[:, sam_valid_mask], reduction='mean')

        '''constractive learning'''
        feature = outputs["feature"]
        selected_pairs = sampling_pairs_in_mask(gt_mask, 800)
        selected_points = sampling_in_mask(gt_mask, 1000)
        fea_loss = 0
        for i in range(len(selected_pairs)):
            f1 = feature[selected_pairs[i][0][:,0], selected_pairs[i][0][:,1]]
            f2 = feature[selected_pairs[i][1][:,0], selected_pairs[i][1][:,1]]
            fea_loss += cosine_similarity_loss(f1.permute(1, 0), f2.permute(1, 0))
        fea_loss = fea_loss / len(selected_pairs)
        fea_up = self.fea_up(feature[selected_points[:,0], selected_points[:,1], :]).permute(1,0)
        up_loss = cosine_similarity_loss(fea_up, gt_fea[:, selected_points[:,0] , selected_points[:,1]])
        
        if self.step % 10 == 0:
            # Before, we made split sh and colors onto different optimizer, with shs having a low learning rate
            # This is slow, instead we apply a regularization every few steps
            sh_reg = self.colors_all[:, 1:, :].norm(dim=1).mean()
            scale_exp = torch.exp(self.scales)
            scale_reg = torch.maximum(scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1), torch.tensor(self.config.max_gauss_ratio)) - self.config.max_gauss_ratio
            scale_reg = 0.1 * scale_reg.mean()
        else:
            sh_reg = torch.tensor(0.0).to(self.device)
            scale_reg = torch.tensor(0.0).to(self.device)
            
        return {"main_loss": (1 - self.config.ssim_lambda) * Ll1 + self.config.ssim_lambda * simloss,
                "feature_loss": fea_loss, "up_loss": up_loss, "depth_loss": depth_loss, "normal_loss": normal_loss, 
                "sh_reg": sh_reg, "scale_reg":scale_reg}
        # return {"main_loss": (1 - self.config.ssim_lambda) * Ll1 + self.config.ssim_lambda * simloss,
        #         "feature_loss": fea_loss, "depth_loss": depth_loss, "normal_loss": normal_loss, 
        #         "sh_reg": sh_reg, "scale_reg":scale_reg}

    @torch.no_grad()
    def get_outputs_for_camera(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Overridden for a camera-based gaussian model.

        Args:
            camera: generates raybundle
        """
        assert camera is not None, "must provide camera to gaussian model"
        self.set_crop(obb_box)
        outs = self.get_outputs(camera.to(self.device))
        return outs  # type: ignore

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Writes the test image outputs.

        Args:
            image_idx: Index of the image.
            step: Current step.
            batch: Batch of data.
            outputs: Outputs of the model.

        Returns:
            A dictionary of metrics.
        """
        with torch.no_grad():
            d = self._get_downscale_factor()
            if d > 1:
                newsize = [batch["image"].shape[0] // d, batch["image"].shape[1] // d]
                gt_img = TF.resize(batch["image"].permute(2, 0, 1), newsize, antialias=None).permute(1, 2, 0)
                predicted_rgb = TF.resize(outputs["rgb"].permute(2, 0, 1), newsize, antialias=None).permute(1, 2, 0)
            else:
                gt_img = batch["image"]
                predicted_rgb = outputs["rgb"]

            gt_rgb = gt_img.to(self.device)

            combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)

            gt_normal = batch["normal"].permute(2, 0, 1).unsqueeze(0).to(self.device)
            gt_normal = F.interpolate(gt_normal, size=(gt_img.shape[0], gt_img.shape[1]), mode='bilinear').squeeze(0)
            gt_depth = batch["depth"].permute(2, 0, 1).unsqueeze(0).to(self.device)
            depth_mask = (gt_depth > 0.1) * 1.0
            gt_depth = F.interpolate(gt_depth, size=(gt_img.shape[0], gt_img.shape[1]), mode='bilinear').squeeze(0)
            depth_mask = F.interpolate(depth_mask, size=(gt_img.shape[0], gt_img.shape[1]), mode='nearest').squeeze(0)
            gt_mask = batch["sam_mask"].to(self.device)
            gt_mask = F.interpolate(gt_mask.float().unsqueeze(0).unsqueeze(0), size=(gt_img.shape[0], gt_img.shape[1]), 
                                    mode='nearest').squeeze(0).squeeze(0)
            valid_mask = batch["valid_mask"].to(self.device)
            valid_mask = F.interpolate(valid_mask.float().unsqueeze(0).unsqueeze(0), size=(gt_img.shape[0], gt_img.shape[1]), 
                                    mode='nearest').squeeze(0).squeeze(0)
            depth_mask = depth_mask * valid_mask
            depth_mask = depth_mask > 0
            valid_mask = valid_mask > 0
            gt_mask[~valid_mask] = -1.0
            gt_fea = batch["feature"].permute(2, 0, 1).float().to(self.device)
            gt_fea = F.interpolate(gt_fea.unsqueeze(0), size=(gt_img.shape[0], gt_img.shape[1]), mode='nearest').squeeze(0)

            # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
            gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
            predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]
            normal = outputs["normal"].permute(2,0,1)
            depth = outputs["depth"].permute(2, 0, 1)
            normal_loss = 0.5 * F.mse_loss(normal[:, depth_mask[0]], gt_normal[:, depth_mask[0]], reduction='mean') + 0.5 * cosine_similarity_loss(normal[:, depth_mask[0]], gt_normal[:, depth_mask[0]])
            depth_loss = F.l1_loss(depth[depth_mask], gt_depth[depth_mask], reduction='mean')
            '''pixel level'''
            # fea_out = self.fea_up(outputs["feature"])
            # fea_out = outputs["feature"]
            # predicted_fea = fea_out.permute(2, 0, 1)
            # fea_loss = cosine_similarity_loss(predicted_fea, gt_fea)
            # fea_loss = cosine_similarity_loss(predicted_fea[:, valid_mask], gt_fea[:, valid_mask])
            '''low rank decomposition'''
            # feature = F.normalize(outputs["feature"][:,:,:-6], dim=-1)
            # fea_up, fea_down, normal = self.fea_up(gt_fea.permute(1, 2, 0), outputs["feature"][:,:,-6:-3], selected_points)
            # up_loss = 0.5 * cosine_similarity_loss(fea_up, gt_fea[:, selected_points[:,0] , selected_points[:,1]]) \
            #         + 0.5 * F.mse_loss(fea_up, gt_fea[:, selected_points[:,0] , selected_points[:,1]], reduction='mean')
            # fea_loss = 0.5 * cosine_similarity_loss(feature.permute(2, 0, 1)[:, valid_mask], fea_down[:, valid_mask]) \
            #         + 0.5 * F.mse_loss(feature.permute(2, 0, 1)[:, valid_mask], fea_down[:, valid_mask], reduction='mean')
            '''constractive learning'''
            feature = outputs["feature"]
            selected_pairs = sampling_pairs_in_mask(gt_mask, 800)
            selected_points = sampling_in_mask(gt_mask, 1000)
            fea_loss = 0
            for i in range(len(selected_pairs)):
                f1 = feature[selected_pairs[i][0][:,0], selected_pairs[i][0][:,1]]
                f2 = feature[selected_pairs[i][1][:,0], selected_pairs[i][1][:,1]]
                fea_loss += cosine_similarity_loss(f1.permute(1, 0), f2.permute(1, 0))
            fea_loss = fea_loss / len(selected_pairs)
            fea_up = self.fea_up(feature[selected_points[:,0], selected_points[:,1], :]).permute(1,0)
            up_loss = cosine_similarity_loss(fea_up, gt_fea[:, selected_points[:,0] , selected_points[:,1]])
    

            psnr = self.psnr(gt_rgb, predicted_rgb)
            ssim = self.ssim(gt_rgb, predicted_rgb)
            lpips = self.lpips(gt_rgb, predicted_rgb)

            # all of these metrics will be logged as scalars
            metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
            metrics_dict["lpips"] = float(lpips)
            metrics_dict["feasim"] = float(fea_loss.item())
            metrics_dict["normal"] = float(normal_loss.item())
            metrics_dict["depth"] = float(depth_loss.item())
            metrics_dict["up_dist"] = float(up_loss.item())

            images_dict = {"img": combined_rgb}

        return metrics_dict, images_dict
