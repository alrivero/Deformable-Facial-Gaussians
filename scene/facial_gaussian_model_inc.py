import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from ..flame.flame import FlameHead


class FacialGaussianModel:
    def __init__(self, sh_degree, shape_len=300, exp_len=100):

        # Idential init, but we're differentiating from facial vertex information
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree

        self._global_xyz = torch.empty(0)
        self._global_features_dc = torch.empty(0)
        self._global_features_rest = torch.empty(0)
        self._global_scaling = torch.empty(0)
        self._global_rotation = torch.empty(0)
        self._global_opacity = torch.empty(0)
        self.global_max_radii2D = torch.empty(0)
        self.global_xyz_gradient_accum = torch.empty(0)

        self.optimizer = None

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        # Now, our facial model (FLAME) and its atttributes
        self.flame_head = FlameHead(shape_len, exp_len).cuda()

        self._flame_xyz = torch.empty(0)
        self._flame_features_dc = torch.empty(0)
        self._flame_features_rest = torch.empty(0)
        self._flame_scaling = torch.empty(0)
        self._flame_rotation = torch.empty(0)
        self._flame_opacity = torch.empty(0)
        self.flame_max_radii2D = torch.empty(0)
        self.flame_xyz_gradient_accum = torch.empty(0)

        self.trained_flame_params = {}
        self.split_param_groups = ["xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation"]

    @property
    def _xyz(self):
        return torch.cat((self._flame_xyz, self._global_xyz))
    
    @_xyz.setter
    def _xyz(self, value):
        self._flame_xyz = value.data[:len(self._flame_xyz)] # NOT A PARAM
        self._global_xyz = nn.Parameter(value.data[len(self._flame_xyz):])
    
    @property
    def _features_dc(self):
        return torch.cat((self._flame_features_dc, self._global_features_dc))
    
    @_features_dc.setter
    def _features_dc(self, value):
        self._flame_features_dc = nn.Parameter(value.data[:len(self._flame_features_dc)])
        self._global_features_dc = nn.Parameter(value.data[len(self._flame_features_dc):])
    
    @property
    def _features_rest(self):
        return torch.cat((self._flame_features_rest, self._global_features_rest))
    
    @_features_rest.setter
    def _features_rest(self, value):
        self._flame_features_rest = nn.Parameter(value.data[:len(self._flame_features_rest)])
        self._global_features_rest = nn.Parameter(value.data[len(self._flame_features_rest):])
    
    @property
    def _scaling(self):
        return torch.cat((self._flame_scaling, self._global_scaling))
    
    @_scaling.setter
    def _scaling(self, value):
        self._flame_scaling = nn.Parameter(value.data[:len(self._flame_scaling)])
        self._global_scaling = nn.Parameter(value.data[len(self._flame_scaling):])
    
    @property
    def _rotation(self):
        return torch.cat((self._flame_rotation, self._global_rotation))
    
    @_rotation.setter
    def _rotation(self, value):
        self._flame_rotation = nn.Parameter(value.data[:len(self._flame_rotation)])
        self._global_rotation = nn.Parameter(value.data[len(self._flame_rotation):])
    
    @property
    def _opacity(self):
        return torch.cat((self._flame_opacity, self._global_opacity))
    
    @_opacity.setter
    def _opacity(self, value):
        self._flame_opacity = nn.Parameter(value.data[:len(self._flame_opacity)])
        self._global_opacity = nn.Parameter(value.data[len(self._flame_opacity):])
    
    @property
    def max_radii2D(self):
        return torch.cat((self.flame_max_radii2D, self.global_max_radii2D))
    
    @max_radii2D.setter
    def max_radii2D(self, value):
        self._flame_max_radii2D = nn.Parameter(value.data[:len(self._flame_max_radii2D)])
        self._global_max_radii2D = nn.Parameter(value.data[len(self._flame_max_radii2D):])
    
    @property
    def xyz_gradient_accum(self):
        return torch.cat((self.flame_xyz_gradient_accum, self.global_xyz_gradient_accum))
    
    @xyz_gradient_accum.setter
    def xyz_gradient_accum(self, value):
        self._flame_xyz_gradient_accum = nn.Parameter(value.data[:len(self._flame_xyz_gradient_accum)])
        self._global_xyz_gradient_accum = nn.Parameter(value.data[len(self._flame_xyz_gradient_accum):])

    def flame_forward(self, params, t):
        final_params = {}
        for param_name in params.keys():
            if param_name in self.trained_flame_params.keys():
                if param_name == "scale" or param_name == "shape":
                    final_params[param_name] = self.trained_flame_params[param_name]
                else:
                    final_params[param_name] = self.trained_flame_params[param_name][t]
            else:
                final_params[param_name] = params[param_name]

        vertices, landmarks = self.flame_head(
            final_params["shape"][None],
            final_params["expr"][None],
            final_params["rotation"][None],
            final_params["neck_pose"][None],
            final_params["jaw_pose"][None],
            final_params["eyes_pose"][None],
            final_params["translation"][None]
        )

        vertices *= final_params["scale"]
        landmarks *= final_params["scale"]

        return vertices, landmarks
    
    def scale_prism(self, min_values, max_values, scale_factors):
        """ Scale the min and max values of a prism by a given scale factor. """
        center = (min_values + max_values) / 2
        half_size_scaled = ((max_values - min_values) * scale_factors) / 2
        new_min = center - half_size_scaled
        new_max = center + half_size_scaled
        return new_min, new_max
    
    def define_flame_global_local_neighborhood(self, scale_factors=[1.1, 1.1, 2.2]):
        with torch.no_grad():
            max_x = self._flame_xyz[:, 0].max()
            max_y = self._flame_xyz[:, 1].max()
            max_z = self._flame_xyz[:, 2].max()
            min_x = self._flame_xyz[:, 0].min()
            min_y = self._flame_xyz[:, 1].min()
            min_z = self._flame_xyz[:, 2].min()

            min_values = torch.tensor([min_x, min_y, min_z])
            max_values = torch.tensor([max_x, max_y, max_z])

            scale_factors = torch.tensor(scale_factors)
            scaled_min, scaled_max = self.scale_prism(min_values, max_values, scale_factors)

            mask_x = (self._global_xyz[:, 0] >= 0) & (self._global_xyz[:, 0] <= scaled_max[0])
            mask_y = (self._global_xyz[:, 1] >= scaled_min[1]) & (self._global_xyz[:, 1] <= scaled_max[1])
            mask_z = (self._global_xyz[:, 2] >= scaled_min[2]) & (self._global_xyz[:, 2] <= scaled_max[2])
            combined_mask = mask_x & mask_y & mask_z

            return combined_mask
        
    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float, cull_z=5):
        self.spatial_lr_scale = 5
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())

        valid_points = (fused_point_cloud[:, 2] > cull_z).cpu().numpy()
        fused_point_cloud = fused_point_cloud[valid_points]
        fused_color = fused_color[valid_points]

        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points[valid_points])).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._global_xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._global_features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._global_features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._global_scaling = nn.Parameter(scales.requires_grad_(True))
        self._global_rotation = nn.Parameter(rots.requires_grad_(True))
        self._global_opacity = nn.Parameter(opacities.requires_grad_(True))
        self.global_max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
    
    def init_flame_mesh(self, flame_params, cull_global_neighborhood=True):
        mesh_vertices, _ = self.flame_forward(flame_params)
        features = torch.zeros((mesh_vertices.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()

        print("Number of mesh points at initialisation : ", mesh_vertices.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(mesh_vertices)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((mesh_vertices.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((mesh_vertices.shape[0], 1), dtype=torch.float, device="cuda"))

        self._flame_xyz = mesh_vertices # These change frame-to-frame and are not a param
        self._flame_features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._flame_features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._flame_scaling = nn.Parameter(scales.requires_grad_(True))
        self._flame_rotation = nn.Parameter(rots.requires_grad_(True))
        self._flame_opacity = nn.Parameter(opacities.requires_grad_(True))
        self.flame_max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        # Also include rotation and translation and scale as params that can be adjusted for the flame mesh
        self.trained_flame_params["rotation"] = nn.Parameter(torch.tensor(flame_params["rotation"]).float())
        self.trained_flame_params["translation"] = nn.Parameter(torch.tensor(flame_params["translation"]).float())
        self.trained_flame_params["scale"] = nn.Parameter(torch.tensor(flame_params["scale"]).float())

        if cull_global_neighborhood:
            valid_points = ~self.define_flame_global_local_neighborhood()

            self._global_xyz = nn.Parameter(self._global_xyz.data[valid_points])
            self._global_features_dc = nn.Parameter(self._global_features_dc.data[valid_points])
            self._global_features_rest = nn.Parameter(self._global_features_rest.data[valid_points])
            self._global_scaling = nn.Parameter(self._global_scaling.data[valid_points])
            self._global_rotation = nn.Parameter(self._global_rotation.data[valid_points])
            self._global_opacity = nn.Parameter(self._global_opacity.data[valid_points])
            self._global_max_radii2D = nn.Parameter(self._global_max_radii2D.data[valid_points])

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        self.spatial_lr_scale = 5

        l = [
            {'params': [self._global_xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "global_xyz"},
            {'params': [self._global_features_dc], 'lr': training_args.feature_lr, "name": "global_f_dc"},
            {'params': [self._global_features_rest], 'lr': training_args.feature_lr / 20.0, "name": "global_f_rest"},
            {'params': [self._global_opacity], 'lr': training_args.opacity_lr, "name": "global_opacity"},
            {'params': [self._global_scaling], 'lr': training_args.scaling_lr * self.spatial_lr_scale, "name": "global_scaling"},
            {'params': [self._global_rotation], 'lr': training_args.rotation_lr, "name": "global_rotation"},
            {'params': [self._flame_features_dc], 'lr': training_args.feature_lr, "name": "flame_f_dc"},
            {'params': [self._flame_features_rest], 'lr': training_args.feature_lr / 20.0, "name": "flame_f_rest"},
            {'params': [self._flame_opacity], 'lr': training_args.opacity_lr, "name": "flame_opacity"},
            {'params': [self._flame_scaling], 'lr': training_args.scaling_lr * self.spatial_lr_scale, "name": "flame_scaling"},
            {'params': [self._flame_rotation], 'lr': training_args.rotation_lr, "name": "flame_rotation"}
        ]

        for flame_param in self.trained_flame_params.keys():
            l.append({'params': [self.trained_flame_params[flame_param]], 'lr': training_args.flame_param_lr, "name": f"flame_{flame_param}_param"})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"].contains("xyz"):
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def replace_tensor_to_optimizer(self, tensor, name):
        if name in self.split_param_groups:
            global_param = super().replace_tensor_to_optimizer(self, tensor, f"global_{name}")
            flame_param = super().replace_tensor_to_optimizer(self, tensor, f"flame_{name}")

            if name == "xyz":
                (None, global_param[name])
            else:
                return (flame_param[name], global_param[name])
        else:
            return super().replace_tensor_to_optimizer(self, tensor, name)
        
    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._flame_opacity = optimizable_tensors[0]
        self._global_opacity = optimizable_tensors[0]

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"].contains("flame"): # Do not remove flame data
                continue
            elif group["name"].contains("global"):
                group_mask = mask[len(self._flame_xyz):]
            else:
                group_mask = mask

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][group_mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][group_mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][group_mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][group_mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        valid_points_mask[:len(self._gloval_xyz)] = True # Never prune any FLAME mesh points
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._flame_features_dc = optimizable_tensors["flame_f_dc"]
        self._flame_features_rest = optimizable_tensors["flame_f_rest"]
        self._flame_opacity = optimizable_tensors["flame_opacity"]
        self._flame_scaling = optimizable_tensors["flame_scaling"]
        self._flame_rotation = optimizable_tensors["flame_rotation"]
        self._global_xyz = optimizable_tensors["global_xyz"]
        self._global_features_dc = optimizable_tensors["global_f_dc"]
        self._global_features_rest = optimizable_tensors["global_f_rest"]
        self._global_opacity = optimizable_tensors["global_opacity"]
        self._global_scaling = optimizable_tensors["global_scaling"]
        self._global_rotation = optimizable_tensors["global_rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"].contains("flame"): # Do not augment flame data
                continue

            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group['name']]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                    dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                                                       dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors
    
    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                              new_rotation):
        d = {"global_xyz": new_xyz,
             "global_f_dc": new_features_dc,
             "global_f_rest": new_features_rest,
             "global_opacity": new_opacities,
             "global_scaling": new_scaling,
             "global_rotation": new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._global_xyz = optimizable_tensors["global_xyz"]
        self._global_features_dc = optimizable_tensors["global_f_dc"]
        self._global_features_rest = optimizable_tensors["global_f_rest"]
        self._global_opacity = optimizable_tensors["global_opacity"]
        self._global_scaling = optimizable_tensors["global_scaling"]
        self._global_rotation = optimizable_tensors["global_rotation"]

        self.gxyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")



