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
        super.__init__(sh_degree)

        # Our facial model (FLAME) and its atttributes
        self.flame_head = FlameHead(shape_len, exp_len).cuda()
    
    def flame_forward(self, params):
        vertices, landmarks = self.flame_head(
            params["flame_shape"][None],
            params["flame_expr"][None],
            params["flame_rotation"][None],
            params["flame_neck_pose"][None],
            params["flame_jaw_pose"][None],
            params["flame_eyes_pose"][None],
            params["flame_translation"][None]
        )

        vertices *= params["flame_scale"]
        landmarks *= params["flame_scale"]

        return vertices, landmarks

    