import json
import math
import os
import time
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union


import imageio
import nerfview
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import yaml
from datasets.colmap import Dataset, Parser
from datasets.traj import (
    generate_interpolated_path,
    generate_ellipse_path_z,
    generate_spiral_path,
)
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from fused_ssim import fused_ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never
from utils import AppearanceOptModule, CameraOptModule, knn, rgb_to_sh, set_random_seed
from lib_bilagrid import (
    BilateralGrid,
    slice,
    color_correct,
    total_variation_loss,
)

from gsplat.compression import PngCompression
from gsplat.distributed import cli
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat.optimizers import SelectiveAdam

from gsplat import _rasterization, _radar_rasterization #, rasterization_radar2d

from gsplat.rendering import azimuth_antenna_gain_projection, spectral_leakage

import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(parent_dir)
from radar.dataset.dataloader import WaveSensorDataParser, WaveSensorDataset

from radar.utils import polar_to_cart, cart_to_polar
import wandb
from datetime import datetime
from matplotlib.colors import Normalize
from radar.utils import visualize_in_polar_space, visualize_in_cart_space, visualize_in_cart_space_separated, visualize_signal_decomposition, visualize_signal_refl, visualize_with_lidar
from radar.eval_utils import eval_geometry 

from radar.utils import get_multipath_model

import pickle
@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = False
    # Path to the .pt files. If provide, it will skip training and run evaluation only.
    ckpt: Optional[List[str]] = None
    # Name of compression strategy to use
    compression: Optional[Literal["png"]] = None
    # Render trajectory path
    render_traj_path: str = "interp"

    seq_name: str = "boreas-2021-09-02-11-42"

    # Path to the Mip-NeRF 360 dataset
    data_dir: str = "data/360_v2/garden"
    # Downsample factor for the dataset
    data_factor: int = 4
    # Directory to save results
    result_dir: str = "results/radar_test"
    # Every N images there is a test image
    test_every: int = 8
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
    # Normalize the world space
    normalize_world_space: bool = False # default to False
    # Camera model
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "ortho" # default to ortho for radar

    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 30_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])

    # Initialization strategy
    init_type: str = "random" # "predefined" or "random"
    # Initial number of GSs. Ignored if using predefined
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using predefined
    init_extent: float = 1.0
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2
    # Weight for max size loss
    maxsize_lambda: float = 100
    # Weight for l1 occ loss
    l1occloss_lambda: float = 0.5
    # Weight for opa and noise regularization
    opa_noise_reg_loss_lambda: float = 100
    # Near plane clipping distance
    near_plane: float = -10
    # Far plane clipping distance
    far_plane: float = 10

    # Strategy for GS densification
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=DefaultStrategy
    )
    
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use visible adam from Taming 3DGS. (experimental)
    visible_adam: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = False

    # Opacity regularization
    opacity_reg: float = 0.0
    # Scale regularization
    scale_reg: float = 0.0

    # Enable camera optimization.
    pose_opt: bool = False
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False
    
    # Wandb
    use_wandb: bool = True
    # Dump information to wandb every this steps
    wandb_every: int = 100
    # Save training images to wandb
    wandb_save_image: bool = True
    wandb_img_every: int = 100

    lpips_net: Literal["vgg", "alex"] = "alex"

    # Scanning radar config
    intermediate_azimuth_resolution: float = 0.1
    max_range: float = None # (m)
    init_scale: float = 0.5 # (m)
    
    frame_selection: Optional[List[int]] = None
    preprocess_thres: bool = True
    spectral_leakage: bool = False
    sinc_width: float = 1.
    multipath_weight: float = 0.6
    multipath_angle_diff_thres_deg: float = 10.
    
    pause_geo_opt_after: int = None

    # Eval config
    # distance tolerance for precision, recall, and accuracy caculation
    dist_tolerance: float = 0.5
    eval_ego_shift: Optional[List[float]] = None
    
    viz_3d: bool = False
    viz_lidar: bool = False

    save_fig: bool = False
    eval_set: Literal["train", "val", "all", "val+all", "all+val"] = "all"
    use_lidar_map: bool = False

    # Ablation
    use_noise_probs: bool = True
    use_polar: bool = True
    radar_map_thres: float = 0.15
    
    synced_lidar_map_name: str = 'synced_lidar_map_win5'
    radar_average_map_name: str = 'radar_average_map_polar/res:0.0596_dist:50_win_size:5_CR_thres:0.21_smooth:3.0' # "baseline_polar_occ_filtered_polar/res:0.0596_dist:50_win_size:0"
    
    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)

        strategy = self.strategy
        if isinstance(strategy, DefaultStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        else:
            assert_never(strategy)


def create_splats_with_optimizers(
    parser: WaveSensorDataParser,
    init_type: str = "predefined",
    init_num_pts: int = 100_000,
    init_extent: float = 1.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    scene_scale: float = 1.0,
    scene_center: np.array = None,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    visible_adam: bool = False,
    batch_size: int = 1,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    if init_type == "predefined":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    elif init_type == "random":
        points = init_extent * scene_scale/2. * (torch.rand((init_num_pts, 3)) * 2 - 1)
        # points = init_extent * 50. * (torch.rand((init_num_pts, 3)) * 2 - 1) # fixed !!!!!

        points += scene_center.astype(np.float32)
        points[:,2] = 0 # only intialize on xy plane in the release version
        rgbs = torch.rand((init_num_pts, 3))
        
        # # override point position
        # points[:,0] = 0.0
        # points[:,1] = 10.0
    else:
        raise ValueError("Please specify a correct init_type: predefined or random")

    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    # dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    # dist_avg = torch.sqrt(dist2_avg)
    # scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

    # Override scale size
    scales = torch.ones_like(points) * cfg.init_scale #0.5
    scales = torch.log(scales * init_scale)
    
    # Distribute the GSs to different ranks (also works for single rank)
    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]

    N = points.shape[0]
    quats = torch.rand((N, 4))  # [N, 4]
    # # override quats
    # quats[...,:] = quats[0,:]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]
    noise_probs = torch.logit(torch.full((N,), init_opacity))  # [N,]

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), 1.6e-4 * scene_scale),
        ("scales", torch.nn.Parameter(scales), 5e-3),
        ("quats", torch.nn.Parameter(quats), 1e-3),
        ("opacities", torch.nn.Parameter(opacities), 5e-2),
        ("noise_probs", torch.nn.Parameter(noise_probs), 5e-2),

    ]

    # color is SH coefficients.
    colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
    colors[:, 0, :] = rgb_to_sh(rgbs)
    params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), 2.5e-3)) # 2.5e-3
    params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), 2.5e-3)) # 2.5e-3 / 20

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    BS = batch_size * world_size
    optimizer_class = None
    if sparse_grad:
        optimizer_class = torch.optim.SparseAdam
    elif visible_adam:
        optimizer_class = SelectiveAdam
    else:
        optimizer_class = torch.optim.Adam
    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            # TODO: check betas logic when BS is larger than 10 betas[0] will be zero.
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, _, lr in params
    }
    return splats, optimizers

class Runner:
    """Engine for training and testing."""

    def __init__(
        self, local_rank: int, world_rank, world_size: int, cfg: Config
    ) -> None:
        set_random_seed(42 + local_rank)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"
        
        # Generate experiment name with timestamp
        self.exp_name = f"{cfg.seq_name}_frame_{self.cfg.frame_selection[0]}_{self.cfg.frame_selection[1]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f'Start experiment: {self.exp_name}')
        
        # Where to load data
        cfg.data_dir += "/"+cfg.seq_name

        # Where to dump results.
        cfg.result_dir += "/"+cfg.seq_name+"/"+self.exp_name
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        self.stats_dir = f"{cfg.result_dir}/stats"
        self.render_dir = f"{cfg.result_dir}/renders"
        if cfg.ckpt is None:
            # Setup output directories.
            os.makedirs(cfg.result_dir, exist_ok=True)
            os.makedirs(self.ckpt_dir, exist_ok=True)
            os.makedirs(self.stats_dir, exist_ok=True)
            os.makedirs(self.render_dir, exist_ok=True)
        
        # Tensorboard
        if cfg.ckpt is None:
            self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # Wandb
        if cfg.use_wandb and cfg.ckpt is None:
            wandb.init(project="radar_gs", name=self.exp_name)
            wandb.config.update(cfg)

        # Load data: Training data should contain initial points and colors.
        self.parser = WaveSensorDataParser(
            data_dir=cfg.data_dir,
            synced_lidar_map_name=cfg.synced_lidar_map_name,
            radar_average_map_name=cfg.radar_average_map_name,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
            test_every=cfg.test_every,
            intermediate_azimuth_resolution=cfg.intermediate_azimuth_resolution,
            max_range=cfg.max_range,
            frame_selection=cfg.frame_selection,
        )

        # overwrite use_polar with cfg
        self.parser.use_polar = self.cfg.use_polar
        print('overwrite use_polar with cfg')

        self.trainset = WaveSensorDataset(
            self.parser,
            split="train",
        )
        self.valset = WaveSensorDataset(self.parser, split="val")
        self.allset = WaveSensorDataset(self.parser, split="all")

        self.scene_center = self.parser.scene_center
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        # Model
        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            scene_scale=self.scene_scale,
            scene_center=self.scene_center,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            visible_adam=cfg.visible_adam,
            batch_size=cfg.batch_size,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
        )
        print("Model initialized. Number of GS:", len(self.splats["means"]))

        # Densification Strategy
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)

        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()
        else:
            assert_never(self.cfg.strategy)

        # Compression Strategy
        self.compression_method = None
        if cfg.compression is not None:
            if cfg.compression == "png":
                self.compression_method = PngCompression()
            else:
                raise ValueError(f"Unknown compression strategy: {cfg.compression}")

        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]
            if world_size > 1:
                self.pose_adjust = DDP(self.pose_adjust)

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)
            if world_size > 1:
                self.pose_perturb = DDP(self.pose_perturb)

        self.bil_grid_optimizers = []


        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self.device)
        elif cfg.lpips_net == "vgg":
            # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = nerfview.Viewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                mode="training",
            )

    def rasterize_splats(
        self,
        radarposes: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        use_polar: bool = False,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        means = self.splats["means"]  # [N, 3]
        
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]
        noise_probs = torch.sigmoid(self.splats["noise_probs"])  # [N,]

        if self.cfg.use_noise_probs==False:
            noise_probs = torch.zeros_like(noise_probs) # overwrite with empty noise_probs for ablation studies

        image_ids = kwargs.pop("image_ids", None)
        colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]
        
        if use_polar:
            height = int(360./cfg.intermediate_azimuth_resolution)

        rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        render_powers, render_occupancy, render_noise_probs, render_opa_refl, render_noise_refl, render_reflectance, info = _radar_rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            noise_probs=noise_probs,
            colors=colors,
            viewmats=radarposes, #torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            # absgrad=(
            #     self.cfg.strategy.absgrad
            #     if isinstance(self.cfg.strategy, DefaultStrategy)
            #     else False
            # ),
            # sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            distributed=self.world_size > 1,
            camera_model=self.cfg.camera_model,
            sph=use_polar,
            **kwargs,
        )
        # render_colors = render_powers.clone().repeat(1, 1, 1, 3)
        if masks is not None:
            render_powers[~masks] = 0
            render_occupancy[~masks] = 0
            render_noise_probs[~masks] = 0
            render_reflectance[~masks]=0
        return render_powers, render_occupancy, render_noise_probs, render_opa_refl, render_noise_refl, render_reflectance, info

    def train(self):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        # Dump cfg.
        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = 0

        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]
        if cfg.pose_opt:
            # pose optimization has a learning rate schedule
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )

        # num_workers = len(self.trainset) if len(self.trainset)<4 else 4
        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))
        for step in pbar:
            if not cfg.disable_viewer:
                while self.viewer.state.status == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            radarposes = radarposes_gt = data["radarpose"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            
            if cfg.preprocess_thres:
                pixels = data["image_thres"].to(device) / 255.0  # [1, H, W, 3]
                pixels_raw = data["image"].to(device) / 255.0
            else:
                pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            
            # Obtain occ map
            pixels_occ = data["preprocess_radar_map_polar"].to(device)
            mask = pixels_occ>=cfg.radar_map_thres
            pixels_occ[mask]=1
            pixels_occ[~mask]=0

            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )

            # Obtain image params
            image_ids = data["image_id"].to(device)
            masks = data["mask"].to(device) if "mask" in data else None  # [1, H, W]
            height, width = pixels.shape[1:3]

            if cfg.pose_noise:
                radarposes = self.pose_perturb(radarposes, image_ids)

            if cfg.pose_opt:
                radarposes = self.pose_adjust(radarposes, image_ids)

            # sh schedule
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)
                
            # forward

            # Cart occupancy rendering !!!
            if self.parser.use_polar==False:
                width, height = 1000, 1000
                range_resolution=0.1
                K = torch.tensor(
                [
                    [1/range_resolution, 0, width/2.], # cart. range resolution
                    [0, 1/range_resolution, height/2.],
                    [0, 0, 1],
                ])
                Ks = K.unsqueeze(0).cuda()

            renders, renders_occupancy, renders_noise_probs, render_opa_refl, render_noise_refl, render_reflectance, info = self.rasterize_splats(
                radarposes=radarposes,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use, # TODO: keep this for future use
                near_plane=cfg.near_plane, # TODO: check if this is needed
                far_plane=cfg.far_plane, # TODO: check if this is needed
                image_ids=image_ids,
                masks=masks,
                use_polar=self.parser.use_polar,
            )
            out_img = renders[0]
            out_occ = renders_occupancy[0]
            out_noise = renders_noise_probs[0]
            
            if self.parser.use_polar and self.cfg.spectral_leakage:
                out_img = spectral_leakage(out_img, self.parser.range_resolution, sinc_width=self.cfg.sinc_width)
                out_occ = spectral_leakage(out_occ, self.parser.range_resolution, sinc_width=self.cfg.sinc_width)
                out_noise = spectral_leakage(out_noise, self.parser.range_resolution, sinc_width=self.cfg.sinc_width)

            if self.parser.use_polar:
                out_img = azimuth_antenna_gain_projection(out_img, new_resolution=self.parser.azimuth_resolution, beamwidth=self.parser.azimuth_beamwidth)
                out_occ = azimuth_antenna_gain_projection(out_occ, new_resolution=self.parser.azimuth_resolution, beamwidth=self.parser.azimuth_beamwidth)
                out_noise = azimuth_antenna_gain_projection(out_noise, new_resolution=self.parser.azimuth_resolution, beamwidth=self.parser.azimuth_beamwidth)

            # Densification
            self.cfg.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )

            # Obtain reconstructed multipath signal
            # TODO: lookup closest multipath_source from the list
            # multipath_sources_pose = radarposes.squeeze()
            # current_pose = radarposes.squeeze()
            # multipath_sources_pose = torch.eye(4)
            # current_pose = torch.eye(4)
            # x_shift, y_shift = 0, 0
            # current_reconstructed_multipath , current_azi_id_list, current_range_id_list, current_source_x, current_source_y = \
            #     get_multipath_model(data["multipath_sources"], x_shift, y_shift, width, self.parser.range_resolution, self.parser.azimuth_resolution, angle_diff_thres_deg=10)

            if self.parser.use_polar:
                current_azi_id_list = data["multipath_sources"]['azi_id_list']
                current_reconstructed_multipath = data["multipath_sources"]["reconstructed_signal"]
                multipath_bg = torch.zeros_like(pixels)
                multipath_bg[:, current_azi_id_list] = current_reconstructed_multipath.to(device).float()
                multipath_bg = multipath_bg.permute(1,2,0)
                # rendered image + pre-computed multipath background 
                out_img = torch.clamp(out_img + multipath_bg * cfg.multipath_weight, min=0, max=1)

            # loss # TODO: squeeze() here is not good for batch rendering
            out_img = out_img.squeeze()
            out_occ = out_occ.squeeze()
            out_noise = out_noise.squeeze()
            pixels = pixels.squeeze()
            pixels_occ = pixels_occ.squeeze()

            min_bin_num = int(2.5/self.parser.range_resolution)
            out_img[:,:min_bin_num]=0
            pixels[:,:min_bin_num]=0

            if cfg.preprocess_thres:
                pixels_raw = pixels_raw.squeeze()
            
            if self.parser.use_polar:
                if self.parser.sensor_type=="sonar": # TODO: use self.parser.azimuth_coverage
                    # For sonar, we only use 130 deg for training
                    out_img_130 = out_img.clone()
                    out_img_130 = out_img_130[:pixels.shape[0],:]

                if self.parser.sensor_type=="sonar": # TODO: use self.parser.azimuth_coverage
                    l1loss = F.l1_loss(out_img_130, pixels)
                else:
                    l1loss = F.l1_loss(out_img, pixels)
                    l1_occ_loss = F.l1_loss(out_occ, pixels_occ)
            # else:
                # TODO:
                # pixels_cart = polar_to_cart(pixels.detach().cpu().numpy(), num_bins_to_show, bin_size, num_azims,
                #                 resolution=1000, noise_floor=None, norm=False)
                # pixels_cart = pixels_cart
            
            max_size_loss = torch.mean(torch.relu(torch.exp(torch.clamp(self.splats['scales'], max=10.0)) - cfg.init_scale*2))
            opa_noise_reg_loss = torch.relu(torch.sigmoid(self.splats['opacities']) + torch.sigmoid(self.splats['noise_probs']) - 1).mean()

            ssimloss = 1.0 - fused_ssim(
                out_img.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0), pixels.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0), padding="valid"
            )

            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda
            loss += l1_occ_loss * cfg.l1occloss_lambda
            loss += max_size_loss * cfg.maxsize_lambda

            if opa_noise_reg_loss!=0 and not torch.isnan(opa_noise_reg_loss):
                loss += opa_noise_reg_loss * cfg.opa_noise_reg_loss_lambda

            if torch.isnan(loss)==True:
                print('[Nan in loss]')
                print(f'l1loss:{l1loss}')
                print(f'ssimloss:{ssimloss}')
                print(f'max_size_loss:{max_size_loss}')
                print(f'l1_occ_loss:{l1_occ_loss}')
                print(f'opa_noise_reg_loss:{opa_noise_reg_loss}')
                sys.exit(0)

            # regularizations
            if cfg.opacity_reg > 0.0:
                loss = (
                    loss
                    + cfg.opacity_reg
                    * torch.abs(torch.sigmoid(self.splats["opacities"])).mean()
                )
            if cfg.scale_reg > 0.0:
                loss = (
                    loss
                    + cfg.scale_reg * torch.abs(torch.exp(self.splats["scales"])).mean()
                )

            loss.backward()

            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "
            if cfg.pose_opt and cfg.pose_noise:
                # monitor the pose error if we inject noise
                pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                desc += f"pose err={pose_err.item():.6f}| "
            pbar.set_description(desc)

            # write images (gt and render)
            # if world_rank == 0 and step % 800 == 0:
            #     canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
            #     canvas = canvas.reshape(-1, *canvas.shape[2:])
            #     imageio.imwrite(
            #         f"{self.render_dir}/train_rank{self.world_rank}.png",
            #         (canvas * 255).astype(np.uint8),
            #     )

            if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                
                self.writer.add_scalar("train/l1loss", l1loss.item() * (1.0 - cfg.ssim_lambda), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item() * cfg.ssim_lambda, step)
                self.writer.add_scalar("train/max_size_loss", max_size_loss.item() * cfg.maxsize_lambda, step)
                
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/average_max_scale", torch.exp(self.splats['scales']).max(dim=1).values.mean(), step)
                self.writer.add_scalar("train/average_opacity", torch.sigmoid(self.splats["opacities"]).mean(), step)
                self.writer.add_scalar("train/average_noise_probs", torch.sigmoid(self.splats["noise_probs"]).mean(), step)
                self.writer.add_scalar("train/average_dist", torch.norm(self.splats["means"][:,:2],dim=1).mean(), step)
                self.writer.add_scalar("train/mem", mem, step)
                if cfg.tb_save_image:
                    canvas = torch.cat([pixels, out_img], dim=0).detach().cpu().numpy()
                    # canvas = canvas.reshape(-1, *canvas.shape[2:])
                    canvas = np.expand_dims(canvas, axis=0)
                    self.writer.add_image("train/render", canvas, step)
                self.writer.flush()
                
            if world_rank == 0 and cfg.wandb_every > 0 and step % cfg.wandb_every == 0 and cfg.use_wandb and cfg.ckpt is None:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                wandb.log({"Loss/loss": loss.item()}, step)
                wandb.log({"Loss/l1loss": l1loss.item()}, step)
                wandb.log({"Loss/ssimloss": ssimloss.item()}, step)
                wandb.log({"Loss/l1_occ_loss": l1_occ_loss.item()}, step)
                wandb.log({"Loss/max_size_loss": max_size_loss.item()}, step)
                wandb.log({"Loss/opa_noise_reg_loss": opa_noise_reg_loss.item()}, step)

                wandb.log({"train/num_GS": len(self.splats["means"])}, step)
                wandb.log({"train/average_max_scale": torch.exp(self.splats['scales']).max(dim=1).values.mean()}, step)
                wandb.log({"train/average_opacity": torch.sigmoid(self.splats["opacities"]).mean()}, step)
                wandb.log({"train/average_noise_probs": torch.sigmoid(self.splats["noise_probs"]).mean()}, step)
                wandb.log({"train/average_dist": torch.norm(self.splats["means"][:,:2],dim=1).mean()}, step)
                wandb.log({"train/mem": mem}, step)
                if cfg.wandb_save_image and step % cfg.wandb_img_every == 0 and self.parser.use_polar:
                    canvas = torch.cat([pixels, out_img], dim=0).detach().cpu().numpy()
                    # canvas = canvas.reshape(-1, *canvas.shape[2:])
                    canvas = np.expand_dims(canvas, axis=0)
                    wandb.log({"Viz/render": wandb.Image(canvas)}, step)
                    
                    # Repalce this with viz function !
                    
                    if self.parser.sensor_type=="sonar": # TODO: use self.parser.azimuth_coverage
                        img = torch.cat((out_img_130, pixels),dim=0)
                    else:
                        img = torch.cat((out_img, pixels),dim=0)
                    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
                    ax.imshow(img.detach().cpu().numpy(), vmin=0.0, vmax=1.0)
                    ax.axis('off')
                    wandb.log({"Viz/render_polar": wandb.Image(fig)}, step)
                    plt.close(fig)
                    
                    # save cart image
                    if self.parser.sensor_type=="sonar": # TODO: use self.parser.azimuth_coverage
                        # For sonar, we pad 130 deg polar image to 360 deg 
                        pixels_360 = torch.zeros_like(out_img)
                        pixels_360[:pixels.shape[0],:] = pixels
                    else:
                        pixels_360 = pixels
                        if cfg.preprocess_thres:
                            pixels_raw_360 = pixels_raw
                    
                    num_bins_to_show = pixels.shape[1]
                    bin_size = self.parser.range_resolution
                    num_azims = int(360/self.parser.azimuth_resolution)
                    pixels_cart = polar_to_cart(pixels_360.detach().cpu().numpy(), num_bins_to_show, bin_size, num_azims,
                            resolution=1000, noise_floor=None, norm=False)
                    out_img_cart = polar_to_cart(out_img.detach().cpu().numpy(), num_bins_to_show, bin_size, num_azims,
                            resolution=1000, noise_floor=None, norm=False)
                    # plt.imshow(out_img_cart)
                    
                    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
                    cart_img = np.hstack((out_img_cart, pixels_cart))
                    ax.imshow(cart_img, vmin=0.0, vmax=1.0)
                    ax.axis('off')
                    wandb.log({"Viz/render_cart": wandb.Image(fig)}, step)
                    plt.close(fig)
                    
                    # Viz lidar
                    fig, ax = plt.subplots(1, 3, figsize=(15, 8), dpi=150)
                    canvas = np.zeros_like(pixels_cart)
                    ax[0].imshow(canvas)
                    width = pixels_cart.shape[0]
                    range_resolution = self.parser.max_range*2/width
                    points = data['synced_lidar'][0].numpy()
                    range_mask = np.linalg.norm(points,axis=1) < self.parser.max_range
                    points = points[range_mask]
                    ax[0].scatter(points[:, 0]/range_resolution + width/2, points[:, 1]/range_resolution + width/2, s=0.01, c=points[:, 2], vmin=-5, vmax=5, cmap='jet')
                    ax[0].axis('off')
                    r = np.linalg.norm(points[:,:2],axis=1)
                    z = points[:,2]
                    phi = np.arctan2(z,r)
                    phi = np.abs(phi)
                    norm = Normalize(vmin=0, vmax=1.8/180*np.pi)
                    normalized_phi = norm(phi)
                    colors = plt.cm.viridis.reversed()(normalized_phi)[:,:3]  # Normalize and apply a colormap (e.g., jet)
                    colors[phi > 1.8/180*np.pi] = (1.0, 0, 0)
                    show_mask = phi < 1.8*2/180*np.pi
                    points = points[show_mask]
                    floor_thres = points[:,2] < 1.
                    points = points[floor_thres]
                    ax[1].imshow(canvas)
                    ax[1].scatter(points[:, 0]/range_resolution + width/2, points[:, 1]/range_resolution + width/2, s=0.01, c=points[:, 2], vmin=-5, vmax=5, cmap='jet')
                    ax[1].axis('off')
                    ax[2].imshow(out_img_cart)
                    ax[2].axis('off')
                    wandb.log({"Viz/radar_vs_lidar": wandb.Image(fig)}, step)
                    plt.close(fig)

                    if cfg.preprocess_thres:
                        pixels_raw_cart = polar_to_cart(pixels_raw_360.detach().cpu().numpy(), num_bins_to_show, bin_size, num_azims,
                                resolution=1000, noise_floor=None, norm=False)
                        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
                        cart_img = np.hstack((pixels_cart, pixels_raw_cart))
                        ax.imshow(cart_img, vmin=0.0, vmax=1.0)
                        ax.axis('off')
                        wandb.log({"Preprocessed/Dynamic threshold": wandb.Image(fig)}, step)
                        plt.close(fig)      

                    out_occ_cart = polar_to_cart(out_occ.detach().cpu().numpy(), num_bins_to_show, bin_size, num_azims,
                            resolution=1000, noise_floor=None, norm=False)

                    pixels_occ_cart = polar_to_cart(pixels_occ.detach().cpu().numpy(), num_bins_to_show, bin_size, num_azims,
                                resolution=1000, noise_floor=None, norm=False)
                    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
                    cart_img = np.hstack((pixels_cart, pixels_occ_cart))
                    ax.imshow(cart_img, vmin=0.0, vmax=1.0)
                    ax.axis('off')
                    wandb.log({"Preprocessed/Radar occ map": wandb.Image(fig)}, step)
                    plt.close(fig)

                    out_noise_cart = polar_to_cart(out_noise.detach().cpu().numpy(), num_bins_to_show, bin_size, num_azims,
                            resolution=1000, noise_floor=None, norm=False)
                    fig, ax = plt.subplots(figsize=(25, 8), dpi=150)
                    cart_img = np.hstack((pixels_occ_cart, out_occ_cart, out_noise_cart, out_img_cart, pixels_cart))
                    ax.imshow(cart_img, vmin=0.0, vmax=1.0)
                    ax.axis('off')
                    wandb.log({"Viz/Radar occ & noise": wandb.Image(fig)}, step)
                    plt.close(fig)

            # save checkpoint before updating the model
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means"]),
                }
                print("Step: ", step, stats)
                with open(
                    f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json",
                    "w",
                ) as f:
                    json.dump(stats, f)
                data = {"step": step, "splats": self.splats.state_dict()}
                if cfg.pose_opt:
                    if world_size > 1:
                        data["pose_adjust"] = self.pose_adjust.module.state_dict()
                    else:
                        data["pose_adjust"] = self.pose_adjust.state_dict()
                torch.save(
                    data, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt"
                )

                # Save cfg to a file
                with open(f"{self.ckpt_dir}/config.pkl", "wb") as f:
                    pickle.dump(self.cfg, f)

            # Turn Gradients into Sparse Tensor before running optimizer
            if cfg.sparse_grad:
                assert cfg.packed, "Sparse gradients only work with packed mode."
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],  # [1, nnz]
                        values=grad[gaussian_ids],  # [nnz, ...]
                        size=self.splats[k].size(),  # [N, ...]
                        is_coalesced=len(Ks) == 1,
                    )

            if cfg.visible_adam:
                gaussian_cnt = self.splats.means.shape[0]
                if cfg.packed:
                    visibility_mask = torch.zeros_like(
                        self.splats["opacities"], dtype=bool
                    )
                    visibility_mask.scatter_(0, info["gaussian_ids"], 1)
                else:
                    visibility_mask = (info["radii"] > 0).any(0)

            if step == cfg.pause_geo_opt_after:
                self.optimizers['means'].param_groups[0]['lr'] = 0.0
                self.optimizers['scales'].param_groups[0]['lr'] = 0.0
                self.optimizers['quats'].param_groups[0]['lr'] = 0.0
            
            # Optimize            
            for optimizer in self.optimizers.values():
                if cfg.visible_adam:
                    optimizer.step(visibility_mask)
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.pose_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.bil_grid_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            # Run post-backward steps after backward and optimizer
            if isinstance(self.cfg.strategy, DefaultStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    packed=cfg.packed,
                )
            elif isinstance(self.cfg.strategy, MCMCStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    lr=schedulers[0].get_last_lr()[0],
                )
            else:
                assert_never(self.cfg.strategy)

            # eval the full set
            if step in [i - 1 for i in cfg.eval_steps]:
                self.eval(step, stage="val", save_fig=True, use_lidar_map=cfg.use_lidar_map)
                # self.eval(step, stage="all", save_fig=True, use_lidar_map=cfg.use_lidar_map)

                # self.render_traj(step)

            # run compression
            if cfg.compression is not None and step in [i - 1 for i in cfg.eval_steps]:
                self.run_compression(step=step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (time.time() - tic)
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)

    @torch.no_grad()
    def eval(self, step: int, stage: str = "val", save_fig=True, use_lidar_map=False):
        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        if stage == 'val':
            loader = torch.utils.data.DataLoader(
                self.valset, batch_size=1, shuffle=False, num_workers=1
            )
        elif stage == 'train':
            loader = torch.utils.data.DataLoader(
                self.trainset, batch_size=1, shuffle=False, num_workers=1
            )
        elif stage == 'all':
            loader = torch.utils.data.DataLoader(
                self.allset, batch_size=1, shuffle=False, num_workers=1
            )
        if cfg.eval_ego_shift is not None:
            stage = f'ego_shift_x:{cfg.eval_ego_shift[0]}_y:{cfg.eval_ego_shift[1]}_{stage}'
        ellipse_time = 0
        metrics = defaultdict(list)
        for i, data in enumerate(loader):

            radarposes = data["radarpose"].to(device)
            multipath_source_pose = radarposes.clone().squeeze()
            if cfg.eval_ego_shift is not None:
                radarposes[:,:2,3] += torch.tensor(np.array(cfg.eval_ego_shift)).to(self.device)

            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            masks = data["mask"].to(device) if "mask" in data else None
            height, width = pixels.shape[1:3]

            pixels_occ = data["preprocess_radar_map_polar"]
            mask = pixels_occ>=cfg.radar_map_thres
            pixels_occ[mask]=1
            pixels_occ[~mask]=0

            # Cart occupancy rendering !!!
            W, H = 1000, 1000
            range_resolution=0.1
            K = torch.tensor(
            [
                [1/range_resolution, 0, W/2.], # cart. range resolution
                [0, 1/range_resolution, H/2.],
                [0, 0, 1],
            ])
            K = K.unsqueeze(0).cuda()
            cart_renders, cart_renders_occupancy, cart_renders_noise_probs, render_opa_refl, render_noise_refl, render_reflectance, info = self.rasterize_splats(
                radarposes=radarposes,
                Ks=K,
                width=W,
                height=H,
                sh_degree=cfg.sh_degree, # TODO: keep this for future use
                near_plane=cfg.near_plane, # TODO: check if this is needed
                far_plane=cfg.far_plane, # TODO: check if this is needed
                # image_ids=image_ids,
                # masks=masks,
                use_polar=False,
            )
            out_occ_cart = cart_renders_occupancy
            # Cart occupancy rendering !!!

            torch.cuda.synchronize()
            tic = time.time()
            # forward
            renders, renders_occupancy, renders_noise_probs, render_opa_refl, render_noise_refl, render_reflectance, info = self.rasterize_splats(
                radarposes=radarposes,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree, # TODO: keep this for future use
                near_plane=cfg.near_plane, # TODO: check if this is needed
                far_plane=cfg.far_plane, # TODO: check if this is needed
                # image_ids=image_ids,
                # masks=masks,
                use_polar=self.parser.use_polar,
            )
            out_img = renders[0]
            out_occ = renders_occupancy[0]
            out_noise = renders_noise_probs[0]
            out_occ_refl = render_opa_refl[0]
            out_noise_refl = render_noise_refl[0]
            out_refl = render_reflectance[0]
            
            if self.cfg.spectral_leakage:
                out_img = spectral_leakage(out_img, self.parser.range_resolution, sinc_width=self.cfg.sinc_width)
                out_occ_wo_spectral_leakage = out_occ.clone()
                out_occ = spectral_leakage(out_occ, self.parser.range_resolution, sinc_width=self.cfg.sinc_width)
                out_noise = spectral_leakage(out_noise, self.parser.range_resolution, sinc_width=self.cfg.sinc_width)
                out_occ_refl = spectral_leakage(out_occ_refl, self.parser.range_resolution, sinc_width=self.cfg.sinc_width)
                out_noise_refl = spectral_leakage(out_noise_refl, self.parser.range_resolution, sinc_width=self.cfg.sinc_width)
                out_refl = spectral_leakage(out_refl, self.parser.range_resolution, sinc_width=self.cfg.sinc_width)

            if self.parser.use_polar:
                out_img = azimuth_antenna_gain_projection(out_img, new_resolution=self.parser.azimuth_resolution, beamwidth=self.parser.azimuth_beamwidth)
                out_occ = azimuth_antenna_gain_projection(out_occ, new_resolution=self.parser.azimuth_resolution, beamwidth=self.parser.azimuth_beamwidth)
                out_noise = azimuth_antenna_gain_projection(out_noise, new_resolution=self.parser.azimuth_resolution, beamwidth=self.parser.azimuth_beamwidth)
                out_occ_refl = azimuth_antenna_gain_projection(out_occ_refl, new_resolution=self.parser.azimuth_resolution, beamwidth=self.parser.azimuth_beamwidth)
                out_noise_refl = azimuth_antenna_gain_projection(out_noise_refl, new_resolution=self.parser.azimuth_resolution, beamwidth=self.parser.azimuth_beamwidth)
                out_refl = azimuth_antenna_gain_projection(out_refl, new_resolution=self.parser.azimuth_resolution, beamwidth=self.parser.azimuth_beamwidth)

                if self.cfg.spectral_leakage:
                    # out_occ_wo_spectral_leakage = out_occ.clone()
                    out_occ_wo_spectral_leakage = azimuth_antenna_gain_projection(out_occ_wo_spectral_leakage, new_resolution=self.parser.azimuth_resolution, beamwidth=self.parser.azimuth_beamwidth)

            # Obtain reconstructed multipath signal
            # TODO: lookup closest multipath_source from the list
            current_azi_id_list = data["multipath_sources"]['azi_id_list']
            current_reconstructed_multipath = data["multipath_sources"]["reconstructed_signal"]
            multipath_bg = torch.zeros_like(pixels)
            multipath_bg[:, current_azi_id_list] = current_reconstructed_multipath.to(device).float()
            multipath_bg = multipath_bg.permute(1,2,0)
            # rendered image + pre-computed multipath background 
            out_img += multipath_bg * cfg.multipath_weight

            torch.cuda.synchronize()
            ellipse_time += time.time() - tic

            out_img = torch.clamp(out_img, 0.0, 1.0)
            out_occ = torch.clamp(out_occ, 0.0, 1.0)

            min_bin_num = int(2.5/self.parser.range_resolution)
            out_img[:,:min_bin_num,:]=0
            pixels[:,:,:min_bin_num]=0

            if self.cfg.spectral_leakage:
                out_occ_wo_spectral_leakage = torch.clamp(out_occ_wo_spectral_leakage, 0.0, 1.0)

            lidar_points = data['synced_lidar_map'][0].numpy() if use_lidar_map else data['synced_lidar'][0].numpy()
            # filter near range points on ego vehicle
            lidar_points = lidar_points[np.linalg.norm(lidar_points,axis=1)>3.5]

            if save_fig:
                os.makedirs(f"{self.render_dir}/{step}/{stage}/polar/", exist_ok=True)
                os.makedirs(f"{self.render_dir}/{step}/{stage}/cart/", exist_ok=True)
                os.makedirs(f"{self.render_dir}/{step}/{stage}/lidar/", exist_ok=True)
                os.makedirs(f"{self.render_dir}/{step}/{stage}/FFT/", exist_ok=True)
                os.makedirs(f"{self.render_dir}/{step}/{stage}/OCC/", exist_ok=True)
                os.makedirs(f"{self.render_dir}/{step}/{stage}/pred_FFT/", exist_ok=True)
                os.makedirs(f"{self.render_dir}/{step}/{stage}/pred_occupancy/", exist_ok=True)
                os.makedirs(f"{self.render_dir}/{step}/{stage}/pred_occupancy_cart/", exist_ok=True)
                os.makedirs(f"{self.render_dir}/{step}/{stage}/pred_occupancy_3D/", exist_ok=True)
                os.makedirs(f"{self.render_dir}/{step}/{stage}/pred_occupancy_3D/wo_sl", exist_ok=True)
                os.makedirs(f"{self.render_dir}/{step}/{stage}/LiDAR_BEV/", exist_ok=True)
                os.makedirs(f"{self.render_dir}/{step}/{stage}/LiDAR_FOV_BEV/", exist_ok=True)
                os.makedirs(f"{self.render_dir}/{step}/{stage}/LiDAR_radar_vis/", exist_ok=True)
                os.makedirs(f"{self.render_dir}/{step}/{stage}/reflectance/", exist_ok=True)
                os.makedirs(f"{self.render_dir}/{step}/{stage}/reflectance_alpha/", exist_ok=True)
                os.makedirs(f"{self.render_dir}/{step}/{stage}/reflectance_eta/", exist_ok=True)


                os.makedirs(f"{self.render_dir}/{step}/{stage}/decomposition/full/", exist_ok=True)
                os.makedirs(f"{self.render_dir}/{step}/{stage}/decomposition/occ/", exist_ok=True)
                os.makedirs(f"{self.render_dir}/{step}/{stage}/decomposition/noise/", exist_ok=True)
                os.makedirs(f"{self.render_dir}/{step}/{stage}/decomposition/multipath_bg_cart/", exist_ok=True)
                
                fig, ax = visualize_in_polar_space(pixels, out_img, 
                                                sensor_type = self.parser.sensor_type,
                                                viz_type='plt')
                plt.savefig(f"{self.render_dir}/{step}/{stage}/polar/{stage}_step{step}_{i:04d}.png")
                plt.close(fig)
                fig, ax = visualize_in_cart_space(pixels, out_img, out_occ,
                                                 sensor_type = self.parser.sensor_type, 
                                                 range_resolution = self.parser.range_resolution, 
                                                 azimuth_resolution = self.parser.azimuth_resolution,
                                                 viz_type='plt')
                plt.savefig(f"{self.render_dir}/{step}/{stage}/cart/{stage}_step{step}_{i:04d}.png")
                plt.close(fig)
                
                canvas = visualize_in_polar_space(pixels, out_img,
                                                sensor_type = self.parser.sensor_type,
                                                viz_type='rgb')
                imageio.imwrite(
                    f"{self.render_dir}/{step}/{stage}/polar/{stage}_step{step}_{i:04d}.png",
                    canvas,
                )
                canvas = visualize_in_cart_space(pixels, out_img, out_occ,
                                                sensor_type = self.parser.sensor_type, 
                                                range_resolution = self.parser.range_resolution, 
                                                azimuth_resolution = self.parser.azimuth_resolution,
                                                viz_type='rgb')
                imageio.imwrite(
                    f"{self.render_dir}/{step}/{stage}/cart/{stage}_step{step}_{i:04d}.png",
                    canvas,
                )
                
                fig, ax = visualize_with_lidar(pixels, out_img, out_occ,
                                            points = lidar_points, 
                                            sensor_type = self.parser.sensor_type, 
                                            range_resolution = self.parser.range_resolution, 
                                            azimuth_resolution = self.parser.azimuth_resolution, 
                                            max_range = self.parser.max_range)
                plt.savefig(f"{self.render_dir}/{step}/{stage}/lidar/{stage}_step{step}_{i:04d}.png")
                plt.close(fig)

                out_refl_alpha_mask = out_refl.clone()
                out_refl_eta_mask = out_refl.clone()

                out_refl_color = visualize_signal_refl(out_refl, self.parser.sensor_type, range_resolution, self.parser.azimuth_resolution)
                out_refl_color.save(f"{self.render_dir}/{step}/{stage}/reflectance/{stage}_step{step}_{i:04d}.png", compress_level=0)

                out_refl_alpha_mask[out_occ<0.5]=0 # occ filter !!!!!!!!
                out_refl_color = visualize_signal_refl(out_refl_alpha_mask, self.parser.sensor_type, range_resolution, self.parser.azimuth_resolution)
                out_refl_color.save(f"{self.render_dir}/{step}/{stage}/reflectance_alpha/{stage}_step{step}_{i:04d}.png", compress_level=0)

                out_refl_eta_mask[out_noise<0.5]=0 # noise filter !!!!!!!!
                out_refl_color = visualize_signal_refl(out_refl_eta_mask, self.parser.sensor_type, range_resolution, self.parser.azimuth_resolution)
                out_refl_color.save(f"{self.render_dir}/{step}/{stage}/reflectance_eta/{stage}_step{step}_{i:04d}.png", compress_level=0)

                out_img_cart_gray, out_occ_cart_gray, out_noise_cart_gray, multipath_bg_cart_gray = visualize_signal_decomposition(out_img, out_occ_refl, out_noise_refl, multipath_bg, self.parser.sensor_type, range_resolution, self.parser.azimuth_resolution)
                out_img_cart_gray.save(f"{self.render_dir}/{step}/{stage}/decomposition/full/{stage}_step{step}_{i:04d}.png", compress_level=0)
                out_occ_cart_gray.save(f"{self.render_dir}/{step}/{stage}/decomposition/occ/{stage}_step{step}_{i:04d}.png", compress_level=0)
                out_noise_cart_gray.save(f"{self.render_dir}/{step}/{stage}/decomposition/noise/{stage}_step{step}_{i:04d}.png", compress_level=0)
                multipath_bg_cart_gray.save(f"{self.render_dir}/{step}/{stage}/decomposition/multipath_bg_cart/{stage}_step{step}_{i:04d}.png", compress_level=0)
                # out_occ out_occ_wo_spectral_leakage
                pixels_cart, pixels_occ_cart, out_img_cart, out_occ_cart_, cart_render_occ, fig_lidar, fig_lidar_fov, fig_lidar_fov_visiable, fig_view3d, ax_view3d = visualize_in_cart_space_separated(pixels, pixels_occ, out_img, out_occ, out_occ_cart, lidar_points,
                                                        sensor_type = self.parser.sensor_type, 
                                                        range_resolution = self.parser.range_resolution, 
                                                        azimuth_resolution = self.parser.azimuth_resolution,
                                                        max_range = self.parser.max_range,
                                                        viz_type='rgb',
                                                        viz_3d=self.cfg.viz_3d,
                                                        viz_lidar=self.cfg.viz_lidar)
                if self.cfg.viz_3d:
                    ax_view3d.view_init(elev=45, azim=0) # azim=-90 (back to front view)
                    ax_view3d.dist = 8
                    fig_view3d.savefig(f"{self.render_dir}/{step}/{stage}/pred_occupancy_3D/{stage}_step{step}_{i:04d}.png", format="png", dpi=150, bbox_inches="tight")
                    ax_view3d.view_init(elev=45, azim=0) # azim=-90 (back to front view)
                    ax_view3d.dist = 4
                    fig_view3d.savefig(f"{self.render_dir}/{step}/{stage}/pred_occupancy_3D/{stage}_step{step}_{i:04d}_zoomin.png", format="png", dpi=150, bbox_inches="tight")
                    ax_view3d.view_init(elev=35, azim=-90) # azim=-90 (back to front view)
                    ax_view3d.dist = 8
                    fig_view3d.savefig(f"{self.render_dir}/{step}/{stage}/pred_occupancy_3D/{stage}_step{step}_{i:04d}_front.png", format="png", dpi=150, bbox_inches="tight")
                    plt.close(fig_view3d)
                                
                pixels_cart.save(f"{self.render_dir}/{step}/{stage}/FFT/{stage}_step{step}_{i:04d}.png", compress_level=0)
                pixels_occ_cart.save(f"{self.render_dir}/{step}/{stage}/OCC/{stage}_step{step}_{i:04d}.png", compress_level=0)
                out_img_cart.save(f"{self.render_dir}/{step}/{stage}/pred_FFT/{stage}_step{step}_{i:04d}.png", compress_level=0)
                out_occ_cart_.save(f"{self.render_dir}/{step}/{stage}/pred_occupancy/{stage}_step{step}_{i:04d}.png", compress_level=0)
                cart_render_occ.save(f"{self.render_dir}/{step}/{stage}/pred_occupancy_cart/{stage}_step{step}_{i:04d}.png", compress_level=0)
                
                if self.cfg.viz_lidar:
                    fig_lidar.savefig(f"{self.render_dir}/{step}/{stage}/LiDAR_BEV/{stage}_step{step}_{i:04d}.png", format="png", dpi=150, bbox_inches="tight")
                    fig_lidar_fov.savefig(f"{self.render_dir}/{step}/{stage}/LiDAR_FOV_BEV/{stage}_step{step}_{i:04d}.png", format="png", dpi=150, bbox_inches="tight")
                    fig_lidar_fov_visiable.savefig(f"{self.render_dir}/{step}/{stage}/LiDAR_radar_vis/{stage}_step{step}_{i:04d}.png", format="png", dpi=150, bbox_inches="tight")
                    plt.close(fig_lidar)
                    plt.close(fig_lidar_fov)
                    plt.close(fig_lidar_fov_visiable)

                if self.cfg.spectral_leakage and self.cfg.viz_3d:
                    pixels_cart, pixels_occ_cart, out_img_cart, out_occ_cart_, cart_render_occ, fig_lidar, fig_lidar_fov, fig_lidar_fov_visiable, fig_view3d, ax_view3d = visualize_in_cart_space_separated(pixels, pixels_occ, out_img, out_occ_wo_spectral_leakage, out_occ_cart, lidar_points,
                                                        sensor_type = self.parser.sensor_type, 
                                                        range_resolution = self.parser.range_resolution, 
                                                        azimuth_resolution = self.parser.azimuth_resolution,
                                                        max_range = self.parser.max_range,
                                                        viz_type='rgb',
                                                        viz_3d=self.cfg.viz_3d)
                    if self.cfg.viz_3d:
                        ax_view3d.view_init(elev=45, azim=0) # azim=-90 (back to front view)
                        ax_view3d.dist = 8
                        fig_view3d.savefig(f"{self.render_dir}/{step}/{stage}/pred_occupancy_3D/wo_sl/{stage}_step{step}_{i:04d}.png", format="png", dpi=150, bbox_inches="tight")
                        ax_view3d.view_init(elev=45, azim=0) # azim=-90 (back to front view)
                        ax_view3d.dist = 4
                        fig_view3d.savefig(f"{self.render_dir}/{step}/{stage}/pred_occupancy_3D/wo_sl/{stage}_step{step}_{i:04d}_zoomin.png", format="png", dpi=150, bbox_inches="tight")
                        ax_view3d.view_init(elev=35, azim=-90) # azim=-90 (back to front view)
                        ax_view3d.dist = 8
                        fig_view3d.savefig(f"{self.render_dir}/{step}/{stage}/pred_occupancy_3D/wo_sl/{stage}_step{step}_{i:04d}_front.png", format="png", dpi=150, bbox_inches="tight")

            if world_rank == 0:
                # pixels_p = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
                pixels_p = pixels.unsqueeze(1).repeat(1, 3, 1, 1)  # [1, 3, H, W]
                colors_p = out_img.squeeze().unsqueeze(0).unsqueeze(1).repeat(1, 3, 1, 1)  # [1, 3, H, W]
                metrics["psnr"].append(self.psnr(colors_p, pixels_p))
                metrics["ssim"].append(self.ssim(colors_p, pixels_p))
                metrics["lpips"].append(self.lpips(colors_p, pixels_p))
                
                # print(f'i:{i} psnr:{metrics["psnr"][-1]}')
                
                if len(current_azi_id_list[0])!=0:
                    multipath_colors_p = colors_p[:,:,:,current_azi_id_list.squeeze()]
                    multipath_pixels_p = pixels_p[:,:,:,current_azi_id_list.squeeze()]
                    metrics["multipath_psnr"].append(self.psnr(multipath_colors_p, multipath_pixels_p))
                
                # Eval geometry
                rmse, cd, rcd, precision, recall, accuracy = eval_geometry(pixels, pixels_occ, out_img, out_occ, lidar_points,
                                sensor_type = self.parser.sensor_type, 
                                range_resolution = self.parser.range_resolution, 
                                azimuth_resolution = self.parser.azimuth_resolution,
                                max_range = self.parser.max_range,
                                tau=self.cfg.dist_tolerance)
                metrics["RMSE"].append(torch.tensor(rmse))
                metrics["CD"].append(torch.tensor(cd))
                metrics["R-CD"].append(torch.tensor(rcd))
                metrics["precision"].append(torch.tensor(precision))
                metrics["recall"].append(torch.tensor(recall))
                metrics["accuracy"].append(torch.tensor(accuracy))
                
                # print(f'i:{i} rmse:{metrics["RMSE"][-1]}')

                # Eval Init Occ Map (pixels_occ) geometry
                rmse, cd, rcd, precision, recall, accuracy = eval_geometry(pixels, pixels_occ, out_img, pixels_occ, lidar_points,
                                sensor_type = self.parser.sensor_type, 
                                range_resolution = self.parser.range_resolution, 
                                azimuth_resolution = self.parser.azimuth_resolution,
                                max_range = self.parser.max_range,
                                tau=self.cfg.dist_tolerance)
                metrics["Occ_RMSE"].append(torch.tensor(rmse))
                metrics["Occ_CD"].append(torch.tensor(cd))
                metrics["Occ_R-CD"].append(torch.tensor(rcd))
                metrics["Occ_precision"].append(torch.tensor(precision))
                metrics["Occ_recall"].append(torch.tensor(recall))
                metrics["Occ_accuracy"].append(torch.tensor(accuracy))

                if self.cfg.spectral_leakage:
                    # Eval out_occ_wo_spectral_leakage
                    rmse, cd, rcd, precision, recall, accuracy = eval_geometry(pixels, pixels_occ, out_img, out_occ_wo_spectral_leakage, lidar_points,
                                    sensor_type = self.parser.sensor_type, 
                                    range_resolution = self.parser.range_resolution, 
                                    azimuth_resolution = self.parser.azimuth_resolution,
                                    max_range = self.parser.max_range,
                                    tau=self.cfg.dist_tolerance)
                    metrics["RMSE_wo_sl"].append(torch.tensor(rmse))
                    metrics["CD_wo_sl"].append(torch.tensor(cd))
                    metrics["R-CD_wo_sl"].append(torch.tensor(rcd))
                    metrics["precision_wo_sl"].append(torch.tensor(precision))
                    metrics["recall_wo_sl"].append(torch.tensor(recall))
                    metrics["accuracy_wo_sl"].append(torch.tensor(accuracy))

        if world_rank == 0:
            ellipse_time /= len(loader)

            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            stats.update(
                {
                    "ellipse_time": ellipse_time,
                    "num_GS": len(self.splats["means"]),
                }
            )
            print(
                f" [Image Metrics] "
                f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                f" [Geometry Metrics] "
                f"RMSE: {stats['RMSE']:.3f}, CD: {stats['CD']:.3f}, R-CD: {stats['R-CD']:.4f}, precision: {stats['precision']:.3f}, recall: {stats['recall']:.3f}, accuracy: {stats['accuracy']:.3f} "
                f"Time: {stats['ellipse_time']:.3f}s/image "
                f"Number of GS: {stats['num_GS']}"
            )

            if 'multipath_psnr' in stats:
                print(
                    f"Multipath PSNR: {stats['multipath_psnr']:.3f} "
                )

            print(
                f" [Occ Geometry Metrics] "
                f"Occ_RMSE: {stats['Occ_RMSE']:.3f}, Occ_CD: {stats['Occ_CD']:.3f}, Occ_R-CD: {stats['Occ_R-CD']:.4f}, Occ_precision: {stats['Occ_precision']:.3f}, Occ_recall: {stats['Occ_recall']:.3f}, Occ_accuracy: {stats['Occ_accuracy']:.3f} "
            )

            if self.cfg.spectral_leakage:
                print(
                    f" [Geometry Metrics w/o Spectral Leakage] "
                    f"RMSE: {stats['RMSE_wo_sl']:.3f}, CD: {stats['CD_wo_sl']:.3f}, R-CD: {stats['R-CD_wo_sl']:.4f}, precision: {stats['precision_wo_sl']:.3f}, recall: {stats['recall_wo_sl']:.3f}, accuracy: {stats['accuracy_wo_sl']:.3f} "
                )
                
            # Save the results into a text file
            formatted_dist_tolerance = str(self.cfg.dist_tolerance).replace('.', '_')
            file_path = f"{self.stats_dir}/{stage}_step:{step:04d}_lidarmap:{use_lidar_map}_tau:{formatted_dist_tolerance}_full.txt"
            with open(file_path, "w") as f:
                for metric, values in metrics.items():
                    f.write(f"{metric.upper()}:\n")
                    f.write(", ".join([f"{v:.4f}" for v in values]) + "\n\n")

            # save stats as json
            with open(f"{self.stats_dir}/{stage}_step:{step:04d}_lidarmap:{use_lidar_map}_tau:{formatted_dist_tolerance}.json", "w") as f:
                json.dump(stats, f)

            if cfg.ckpt is None:
                # save stats to tensorboard
                for k, v in stats.items():
                    self.writer.add_scalar(f"{stage}/{k}", v, step)
                self.writer.flush()

    @torch.no_grad()
    def render_traj(self, step: int):
        """Entry for trajectory rendering."""
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        camtoworlds_all = self.parser.radarposes[5:-5]
        if cfg.render_traj_path == "interp":
            camtoworlds_all = generate_interpolated_path(
                camtoworlds_all, 1
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "ellipse":
            height = camtoworlds_all[:, 2, 3].mean()
            camtoworlds_all = generate_ellipse_path_z(
                camtoworlds_all, height=height
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "spiral":
            camtoworlds_all = generate_spiral_path(
                camtoworlds_all,
                bounds=self.parser.bounds * self.scene_scale,
                spiral_scale_r=self.parser.extconf["spiral_radius_scale"],
            )
        else:
            raise ValueError(
                f"Render trajectory type not supported: {cfg.render_traj_path}"
            )

        camtoworlds_all = np.concatenate(
            [
                camtoworlds_all,
                np.repeat(
                    np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all), axis=0
                ),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        # save to video
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30)
        for i in tqdm.trange(len(camtoworlds_all), desc="Rendering trajectory"):
            camtoworlds = camtoworlds_all[i : i + 1]
            Ks = K[None]
                        
            # forward
            renders, renders_occupancy, renders_noise_probs, render_opa_refl, render_noise_refl, info = self.rasterize_splats(
                radarposes=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree, # TODO: keep this for future use
                near_plane=cfg.near_plane, # TODO: check if this is needed
                far_plane=cfg.far_plane, # TODO: check if this is needed
                # image_ids=image_ids,
                # masks=masks,
                use_polar=self.parser.use_polar,
            )
            out_img = renders[0]
            out_occ = renders_occupancy[0]
            
            if self.cfg.spectral_leakage:
                out_img = spectral_leakage(out_img, self.parser.range_resolution, sinc_width=self.cfg.sinc_width)
                out_occ = spectral_leakage(out_occ, self.parser.range_resolution, sinc_width=self.cfg.sinc_width)
            if self.parser.use_polar:
                out_img = azimuth_antenna_gain_projection(out_img, new_resolution=self.parser.azimuth_resolution, beamwidth=self.parser.azimuth_beamwidth)
                out_occ = azimuth_antenna_gain_projection(out_occ, new_resolution=self.parser.azimuth_resolution, beamwidth=self.parser.azimuth_beamwidth)

            # canvas_list = [out_img]
            out_img = out_img.squeeze()
            num_bins_to_show = out_img.shape[1]
            bin_size = self.parser.range_resolution
            num_azims = int(360/self.parser.azimuth_resolution)
            out_img_cart = polar_to_cart(out_img.detach().cpu().numpy(), num_bins_to_show, bin_size, num_azims,
                            resolution=1000, noise_floor=None, norm=False)
            canvas_list = [torch.tensor(out_img_cart).unsqueeze(-1).to(out_img.device)]

            # renders, _, _ = self.rasterize_splats(
            #     camtoworlds=camtoworlds,
            #     Ks=Ks,
            #     width=width,
            #     height=height,
            #     sh_degree=cfg.sh_degree,
            #     near_plane=cfg.near_plane,
            #     far_plane=cfg.far_plane,
            #     # render_mode="RGB+ED",
            # )  # [1, H, W, 4]
            # colors = torch.clamp(renders[..., 0:3], 0.0, 1.0)  # [1, H, W, 3]
            # depths = renders[..., 3:4]  # [1, H, W, 1]
            # depths = (depths - depths.min()) / (depths.max() - depths.min())
            # canvas_list = [colors, depths.repeat(1, 1, 1, 3)]

            # write images
            canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")

    @torch.no_grad()
    def run_compression(self, step: int):
        """Entry for running compression."""
        print("Running compression...")
        world_rank = self.world_rank

        compress_dir = f"{cfg.result_dir}/compression/rank{world_rank}"
        os.makedirs(compress_dir, exist_ok=True)

        self.compression_method.compress(compress_dir, self.splats)

        # evaluate compression
        splats_c = self.compression_method.decompress(compress_dir)
        for k in splats_c.keys():
            self.splats[k].data = splats_c[k].to(self.device)
        self.eval(step=step, stage="compress")

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: nerfview.CameraState, img_wh: Tuple[int, int]
    ):
        """Callable function for the viewer."""
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        render_colors, _, _ = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=W,
            height=H,
            sh_degree=self.cfg.sh_degree,  # active all SH degrees
            radius_clip=3.0,  # skip GSs that have small image radius (in pixels)
        )  # [1, H, W, 3]
        return render_colors[0].cpu().numpy()


def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    runner = Runner(local_rank, world_rank, world_size, cfg)

    if cfg.ckpt is not None:
        # Update save directory
        updated_exp_name = cfg.ckpt[0].split('/')[-3]
        cfg.result_dir = cfg.result_dir.replace(runner.exp_name, updated_exp_name)
        runner.render_dir = runner.render_dir.replace(runner.exp_name, updated_exp_name)
        runner.stats_dir = runner.stats_dir.replace(runner.exp_name, updated_exp_name)
        
        # Run eval only
        # Load ckpt
        ckpts = [
            torch.load(file, map_location=runner.device, weights_only=True)
            for file in cfg.ckpt
        ]
        for k in runner.splats.keys():
            runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])

        from pathlib import Path
        ckpt_path = Path(cfg.ckpt[0])

        # Load config
        with open(ckpt_path.parent / "config.pkl", "rb") as f:
            loaded_config = pickle.load(f)
            runner.cfg = loaded_config
            runner.cfg.ckpt = cfg.ckpt
            runner.cfg.viz_3d = cfg.viz_3d
            runner.cfg.ckpt = cfg.ckpt
                
        step = ckpts[0]["step"]
        
        if cfg.eval_set=='val+all' or cfg.eval_set=='all+val':
            runner.eval(step=step, stage='val', save_fig=cfg.save_fig, use_lidar_map=cfg.use_lidar_map)
            runner.eval(step=step, stage='all', save_fig=cfg.save_fig, use_lidar_map=cfg.use_lidar_map)
        else:
            runner.eval(step=step, stage=cfg.eval_set, save_fig=cfg.save_fig, use_lidar_map=cfg.use_lidar_map)
        
        # runner.eval(step=step, stage='all', save_fig=False, use_lidar_map=use_lidar_map)

        # use_lidar_map = True
        # runner.eval(step=step, stage='val', save_fig=False, use_lidar_map=use_lidar_map)
        # runner.eval(step=step, stage='all', save_fig=False, use_lidar_map=use_lidar_map)
        
        # runner.render_traj(step=step)
        
        if cfg.compression is not None:
            runner.run_compression(step=step)
    else:
        runner.train()
    
    if cfg.use_wandb and cfg.ckpt is None:
        wandb.finish()

    # if not cfg.disable_viewer:
    #     print("Viewer running... Ctrl+C to exit.")
    #     time.sleep(1000000)


if __name__ == "__main__":
    """
    Usage:

    ```bash
    # Single GPU training
    CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default

    # Distributed training on 4 GPUs: Effectively 4x batch size so run 4x less steps.
    CUDA_VISIBLE_DEVICES=0,1,2,3 python simple_trainer.py default --steps_scaler 0.25

    """

    # Config objects we can choose between.
    # Each is a tuple of (CLI description, config object).
    configs = {
        "default": (
            "Gaussian splatting training using densification heuristics from the original paper.",
            Config(
                strategy=DefaultStrategy(verbose=True),
            ),
        ),
        "mcmc": (
            "Gaussian splatting training using densification from the paper '3D Gaussian Splatting as Markov Chain Monte Carlo'.",
            Config(
                init_opa=0.5,
                init_scale=0.1,
                opacity_reg=0.01,
                scale_reg=0.01,
                strategy=MCMCStrategy(verbose=True),
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)

    # try import extra dependencies
    if cfg.compression == "png":
        try:
            import plas
            import torchpq
        except:
            raise ImportError(
                "To use PNG compression, you need to install "
                "torchpq (instruction at https://github.com/DeMoriarty/TorchPQ?tab=readme-ov-file#install) "
                "and plas (via 'pip install git+https://github.com/fraunhoferhhi/PLAS.git') "
            )

    cli(main, cfg, verbose=True)
