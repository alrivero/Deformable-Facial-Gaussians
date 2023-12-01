#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import json
import os
import random
import torch
import numpy as np
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.facial_gaussian_model import FacialGaussianModel
from scene.deform_model import DeformModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from math import ceil, floor

class Scene:
    gaussians: FacialGaussianModel

    def __init__(self, args: ModelParams, gaussians: FacialGaussianModel, load_iteration=None, shuffle=True,
                 resolution_scales=[1.0], include_flame=True, include_landmarks=True):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "cameras_sphere.npz")):
            print("Found cameras_sphere.npz file, assuming DTU data set!")
            scene_info = sceneLoadTypeCallbacks["DTU"](args.source_path, "cameras_sphere.npz", "cameras_sphere.npz")
        elif os.path.exists(os.path.join(args.source_path, "dataset.json")):
            print("Found dataset.json file, assuming Nerfies data set!")
            scene_info = sceneLoadTypeCallbacks["nerfies"](args.source_path, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "poses_bounds.npy")):
            print("Found calibration_full.json, assuming Neu3D data set!")
            scene_info = sceneLoadTypeCallbacks["plenopticVideo"](args.source_path, args.eval, 24)
        elif os.path.exists(os.path.join(args.source_path, "transforms.json")):
            print("Found calibration_full.json, assuming Dynamic-360 data set!")
            scene_info = sceneLoadTypeCallbacks["dynamic360"](args.source_path)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply"),
                                                                   'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale,
                                                                            args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale,
                                                                           args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                 "point_cloud",
                                                 "iteration_" + str(self.loaded_iter),
                                                 "point_cloud.ply"),
                                    og_number_points=len(scene_info.point_cloud.points))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

        if include_flame:
            self.flame_data = np.load(f"{args.source_path}/tracked_flame_params.npz")
            self.gaussians.init_flame_mesh(self.getFlameParams(0), self.flame_data)

        if include_landmarks:
            self.init_landmarks(args)

    def init_landmarks(self, args):
        landmarks_dir = os.path.join(args.source_path, "landmarks")
        landmark_subdirs = sorted([item for item in os.listdir(landmarks_dir) if os.path.isdir(os.path.join(landmarks_dir, item))])

        self.all_lmk2d = []
        self.all_lmk2d_iris = []
        for subdir in landmark_subdirs:
            with open(os.path.join(landmarks_dir, subdir, "keypoints_static_0000.json"), "r") as f:
                lmks_info = json.load(f)
                lmks_view = lmks_info["people"][0]["face_keypoints_2d"]
                lmks_iris = lmks_info["people"][0].get("iris_keypoints_2d", None)

            lmk2d = (
                torch.from_numpy(np.array(lmks_view)).float()[:204].view(-1, 3)
            )
            # scale coordinates
            lmk2d[:, 2:] = 1.0

            if lmks_iris is not None:
                lmk2d_iris = torch.from_numpy(np.array(lmks_iris)).float()[:204]
                lmk2d_iris = lmk2d_iris.view(-1, 3)[[1, 0]]

            if lmks_iris is not None:
                if torch.sum(lmk2d_iris[:, :2] == -1) > 0:
                    lmk2d_iris[:, 2:] = 0.0
                else:
                    lmk2d_iris[:, 2:] = 1.0

            self.all_lmk2d.append(torch.cat((lmk2d, lmk2d_iris))[None])

        self.all_lmk2d = torch.cat(self.all_lmk2d)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def getFlameParams(self, t):
        flame_params = {}
        if isinstance(t, int):
            flame_params["rotation"] = torch.tensor(self.flame_data["rotation"][t]).cuda()
            flame_params["translation"] = torch.tensor(self.flame_data["translation"][t]).cuda()
            flame_params["neck_pose"] = torch.tensor(self.flame_data["neck_pose"][t]).cuda()
            flame_params["jaw_pose"] = torch.tensor(self.flame_data["jaw_pose"][t]).cuda()
            flame_params["eyes_pose"] = torch.tensor(self.flame_data["eyes_pose"][t]).cuda()
            flame_params["expr"] = torch.tensor(self.flame_data["expr"][t]).cuda()
        else:
            t_after = ceil(t * (len(self.flame_data["rotation"]) - 1))
            t_before = floor(t * (len(self.flame_data["rotation"]) - 1))
            lmd = (t * (len(self.flame_data["rotation"]) - 1)) - t_before

            flame_params["rotation"] = torch.tensor(lmd * self.flame_data["rotation"][t_after] + (1 - lmd) * self.flame_data["rotation"][t_before]).cuda()
            flame_params["translation"] = torch.tensor(lmd * self.flame_data["translation"][t_after] + (1 - lmd) * self.flame_data["translation"][t_before]).cuda()
            flame_params["neck_pose"] = torch.tensor(lmd * self.flame_data["neck_pose"][t_after] + (1 - lmd) * self.flame_data["neck_pose"][t_before]).cuda()
            flame_params["jaw_pose"] = torch.tensor(lmd * self.flame_data["jaw_pose"][t_after] + (1 - lmd) * self.flame_data["jaw_pose"][t_before]).cuda()
            flame_params["eyes_pose"] = torch.tensor(lmd * self.flame_data["eyes_pose"][t_after] + (1 - lmd) * self.flame_data["eyes_pose"][t_before]).cuda()
            flame_params["expr"] = torch.tensor(lmd * self.flame_data["expr"][t_after] + (1 - lmd) * self.flame_data["expr"][t_before]).cuda()
        
        flame_params["shape"] = torch.tensor(self.flame_data["shape"]).cuda()
        flame_params["scale"] = torch.tensor(self.flame_data["scale"]).cuda()


        return flame_params