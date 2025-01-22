import math
import os
import time
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
import tyro
from PIL import Image
from torch import Tensor, optim

from gsplat import rasterization, rasterization_2dgs

from gsplat import _rasterization, _radar_rasterization #, rasterization_radar2d

from gsplat import azimuth_antenna_gain_projection

# Set the seed
seed = 42

# NumPy
np.random.seed(seed)

# PyTorch
torch.manual_seed(seed)

# For GPU (if using CUDA)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # For multi-GPU

# Ensures deterministic behavior in GPU operations
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class SimpleTrainer:
    """Trains random gaussians to fit an image."""

    def __init__(
        self,
        gt_image: Tensor,
        num_points: int = 2000,
    ):
        self.device = torch.device("cuda:0")
        self.gt_image = gt_image.to(device=self.device)
        self.num_points = num_points

        fov_x = math.pi / 2.0
        self.H, self.W = gt_image.shape[0], gt_image.shape[1]
        self.focal = 0.5 * float(self.W) / math.tan(0.5 * fov_x)
        self.img_size = torch.tensor([self.W, self.H, 1], device=self.device)

        self._init_gaussians()

    def _init_gaussians(self):
        """Random gaussians"""
        # bd = 2 (ortho) 100 (pinhole)

        bd = 2
        self.means = bd * (torch.rand(self.num_points, 3, device=self.device) - 0.5)
        
        self.means[:,0] *= 80
        self.means[:,1] *= 80
        self.means[:,2] *= 0.01 # flat and close to z plane

        # *0.1 (ortho) *1 (pinhole)
        self.scales = torch.rand(self.num_points, 3, device=self.device) * 1.
        d = 3
        self.rgbs = torch.rand(self.num_points, d, device=self.device)

        u = torch.rand(self.num_points, 1, device=self.device)
        v = torch.rand(self.num_points, 1, device=self.device)
        w = torch.rand(self.num_points, 1, device=self.device)

        self.quats = torch.cat(
            [
                torch.sqrt(1.0 - u) * torch.sin(2.0 * math.pi * v),
                torch.sqrt(1.0 - u) * torch.cos(2.0 * math.pi * v),
                torch.sqrt(u) * torch.sin(2.0 * math.pi * w),
                torch.sqrt(u) * torch.cos(2.0 * math.pi * w),
            ],
            -1,
        )
        self.opacities = torch.ones((self.num_points), device=self.device)

        self.viewmat = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 8.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=self.device,
        )
        self.background = torch.zeros(d, device=self.device)

        self.means.requires_grad = True
        self.scales.requires_grad = True
        self.quats.requires_grad = True
        self.rgbs.requires_grad = True
        self.opacities.requires_grad = True
        self.viewmat.requires_grad = False

    def train(
        self,
        iterations: int = 1000,
        lr: float = 0.01,
        save_imgs: bool = False,
        model_type: Literal["3dgs", "2dgs", "3dgs_autograd", "radar"] = "3dgs",
        camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole",
    ):
        optimizer = optim.Adam(
            [self.rgbs, self.means, self.scales, self.opacities, self.quats], lr
        )
        mse_loss = torch.nn.MSELoss()
        frames = []
        times = [0] * 2  # rasterization, backward
        K = torch.tensor(
            [
                [self.focal, 0, self.W / 2],
                [0, self.focal, self.H / 2],
                [0, 0, 1],
            ],
            device=self.device,
        )
        
        sph = False

        if model_type == "3dgs":
            rasterize_fnc = rasterization
        elif model_type == "2dgs":
            rasterize_fnc = rasterization_2dgs
        elif model_type == "3dgs_autograd":
            rasterize_fnc = _rasterization
        elif model_type == "radar":
            rasterize_fnc = _radar_rasterization
            range_res = 0.2384 # m (default Cart. resolution: 0.2384 image width: 640)
            K = torch.tensor(
            [
                [1/range_res, 0, self.W/2.],
                [0, 1/range_res, self.H/2.],
                [0, 0, 1],
            ],
            device=self.device,
            )
        elif model_type == "radar_sph":
            rasterize_fnc = _radar_rasterization
            sph = True
            range_res = 0.0596 # m
            intermediate_azimuth_res = 0.1 # deg
            self.H = int(360*(1/intermediate_azimuth_res))
            K = torch.tensor(
            [
                [1/range_res, 0, 0], # range resolution => 0.0596 m 
                [0, 180/torch.pi*(1/intermediate_azimuth_res), self.H/2.], # azimuth resolution => 1/10 deg
                [0, 0, 1],
            ],
            device=self.device,
            )

        # elif model_type == "radar2d":
        #     rasterize_fnc = rasterization_radar2d

        for iter in range(iterations):
            start = time.time()

            renders = rasterize_fnc(
                self.means,
                self.quats / self.quats.norm(dim=-1, keepdim=True),
                self.scales,
                torch.sigmoid(self.opacities),
                torch.sigmoid(self.rgbs),
                self.viewmat[None],
                K[None],
                self.W,
                self.H,
                packed=False,
                # render_mode="D",
                camera_model = camera_model,
                sph=sph,
            )[0]
            out_img = renders[0]
            out_alpha_img = out_img[:,:,0].unsqueeze(-1)

            if sph:
                out_alpha_img = azimuth_antenna_gain_projection(out_alpha_img)
                out_img = out_alpha_img.repeat(1,1,3) # convert to 3 channel rgb image

            # print(out_img.shape)
            torch.cuda.synchronize()
            times[0] += time.time() - start

            loss = mse_loss(out_alpha_img, self.gt_image)
            optimizer.zero_grad()
            start = time.time()
            loss.backward()
            torch.cuda.synchronize()
            times[1] += time.time() - start
            optimizer.step()
            print(f"Iteration {iter + 1}/{iterations}, Loss: {loss.item()}")

            if save_imgs and iter % 5 == 0:
                frames.append((out_img.detach().cpu().numpy() * 255).astype(np.uint8))
        if save_imgs:
            # save them as a gif with PIL
            frames = [Image.fromarray(frame) for frame in frames]
            out_dir = os.path.join(os.getcwd(), "results")
            os.makedirs(out_dir, exist_ok=True)
            frames[0].save(
                f"{out_dir}/training.gif",
                save_all=True,
                append_images=frames[1:],
                optimize=False,
                duration=5,
                loop=0,
            )
        print(f"Total(s):\nRasterization: {times[0]:.3f}, Backward: {times[1]:.3f}")
        print(
            f"Per step(s):\nRasterization: {times[0]/iterations:.5f}, Backward: {times[1]/iterations:.5f}"
        )
    
def image_path_to_tensor(image_path: Path):
    import torchvision.transforms as transforms

    img = Image.open(image_path)
    transform = transforms.ToTensor()
    img_tensor = transform(img).permute(1, 2, 0)[..., :3]
    return img_tensor


def main(
    height: int = 256,
    width: int = 256,
    num_points: int = 100000,
    save_imgs: bool = True,
    img_path: Path = None,
    iterations: int = 1000,
    lr: float = 0.01,
    model_type: Literal["3dgs", "2dgs", "3dgs_autograd", "radar", "radar_sph"] = "3dgs",
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole",
) -> None:

    gt_image = image_path_to_tensor(img_path)

    if model_type == 'radar_sph':
        # Load Boreas radar image
        metadata = gt_image[:,:11] # encoder data for accurate azimuth angle
        gt_image = gt_image[:,11:] # 3360 x 400 (range res: 0.0596 m | azimuth res: 0.9 deg | azimuth beamwidth: 1.8 deg)


    trainer = SimpleTrainer(gt_image=gt_image, num_points=num_points)
    trainer.train(
        iterations=iterations,
        lr=lr,
        save_imgs=save_imgs,
        model_type=model_type,
        camera_model=camera_model,
    )


if __name__ == "__main__":
    tyro.cli(main)
