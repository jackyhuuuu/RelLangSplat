import numpy as np
import torch
import os
import json
from tqdm import tqdm
from os import makedirs
import torchvision
from gaussian_renderer import render
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov, camera_to_world


class Camera:
    def __init__(self, cam):
        Rotation = np.array(cam['rotation'])
        Translation = np.array(cam['position'])
        FovX, FovY = cam['fx'], cam['fy']
        self.image_width = cam['width']
        self.image_height = cam['height']

        self.R, self.T = camera_to_world(Rotation, Translation)
        self.FoVx = focal2fov(FovX, self.image_width)
        self.FoVy = focal2fov(FovY, self.image_height)

        # Compute the projection matrix
        self.projection_matrix = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1).cuda()

        # Compute the world-to-view transformation matrix
        self.world_view_transform = torch.tensor(getWorld2View2(self.R, self.T, np.array([0, 0, 0]), 1.0)).transpose(0, 1).cuda()

        # Compute the full projection transformation
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)

        # Compute the camera center in world coordinates
        self.camera_center = self.world_view_transform.inverse()[3, :3]


def render_custom_view(pipeline : PipelineParams, args):
    dataset = model.extract(args)
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        checkpoint = os.path.join(args.model_path, 'chkpnt30000.pth')
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, args, mode='test')
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    with open(os.path.join(args.model_path, "cameras.json"), 'r') as file:
        cam_json = json.load(file)

    if not args.include_feature:
        name = "rgb"
    else:
        name = "language"

    render_path = os.path.join(args.model_path, name, "ours_None", "renders")
    render_npy_path = os.path.join(args.model_path, name, "ours_None", "renders_npy")

    makedirs(render_path, exist_ok=True)
    makedirs(render_npy_path, exist_ok=True)

    for idx, cam in enumerate(tqdm(cam_json, desc="Rendering progress")):
        custom_cam = Camera(cam)
        rendering = render(custom_cam, gaussians, pipeline.extract(args), background, args)
        if name == "rgb":
            rendering = rendering["render"].detach().cpu()
        else:
            rendering = rendering["language_feature_image"].detach().cpu()

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        np.save(os.path.join(render_npy_path, '{0:05d}'.format(idx) + ".npy"), rendering.permute(1, 2, 0).numpy())


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--include_feature", action="store_true")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)

    render_custom_view(pipeline, args)
