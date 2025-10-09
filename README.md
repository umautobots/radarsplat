# RadarSplat

### RadarSplat: Radar Gaussian Splatting for High-Fidelity Data Synthesis and 3D Reconstruction of Autonomous Driving Scenes

Pou-Chun Kung, Skanda Harisha, Ram Vasudevan, Aline Eid, Katherine A. Skinner

[[Paper](https://arxiv.org/pdf/2506.01379)] |
[[Project Page](https://umautobots.github.io/radarsplat)]

<p align="center">
  <img src="assets/radarsplat_teaser.png" alt="RadarSplat Teaser" width="100%" />
</p>

<details>
<summary> <b>Abstract (click to expand)</b> </summary>

High-Fidelity 3D scene reconstruction plays a crucial role in autonomous driving by enabling novel data generation from existing datasets. This allows simulating safety-critical scenarios and augmenting training datasets without incurring further data collection costs. While recent advances in radiance fields have demonstrated promising results in 3D reconstruction and sensor data synthesis using cameras and LiDAR, their potential for radar remains largely unexplored. Radar is crucial for autonomous driving due to its robustness in adverse weather conditions like rain, fog, and snow, where optical sensors often struggle. Although the state-of-the-art radar-based neural representation shows promise for 3D driving scene reconstruction, it performs poorly in scenarios with significant radar noise, including receiver saturation and multipath reflection. Moreover, it is limited to synthesizing preprocessed, noise-excluded radar images, failing to address realistic radar data synthesis. To address these limitations, this paper proposes RadarSplat, which integrates Gaussian Splatting with novel radar noise modeling to enable realistic radar data synthesis and enhanced 3D reconstruction. Compared to the state-of-the-art, RadarSplat achieves superior radar image synthesis (+3.4 PSNR / 2.6x SSIM) and improved geometric reconstruction (-40% RMSE / 1.5x Accuracy), demonstrating its effectiveness in generating high-fidelity radar data and scene reconstruction.

</details>


## Prepare Environment

```bash
conda ...
```

## Prepare Dataset
Download data from [Boreas Dataset](https://www.boreas.utias.utoronto.ca/#/download).
In the paper, we choose:

| Sequence Name             | Weather/Condition |
|-------------------------- |------------------|
| boreas-2021-09-02-11-42   | Sunny            |
| boreas-2021-01-26-11-22   | Snow             |
| boreas-2021-04-29-15-55   | Rain             |
| boreas-2021-09-14-20-00   | Night            |
| boreas-2021-04-08-12-44   | Sunny 2          |


If you want to test on other sequences, please make sure the selected sequences are not part of the [odom_test](https://github.com/utiasASRL/pyboreas/blob/eeb2b1bb302f5386eff7198858846f6d89675fda/pyboreas/data/splits.py#L42-L56) set, so that the ground truth poses are provided.


## Data Preprocessing

Install [pyboreas library](https://github.com/utiasASRL/pyboreas) for data preprocessing.
```bash
pip install asrl-pyboreas
```

Disable local CUDA in case it conflict with CUDA version in conda env
```bash
# Backup and remove from PATH
export PATH=$(echo $PATH | tr ':' '\n' | grep -v '/usr/local/cuda' | paste -sd ':' -)
# Remove from LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -v '/usr/local/cuda' | paste -sd ':' -)
# Unset CUDA_HOME if it points to system CUDA
unset CUDA_HOME

conda install -c nvidia cuda-nvcc=11.7
conda install -c nvidia cuda-toolkit=11.7
```

Then install:
```bash
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

cd examples
pip install -r requirements.txt
```

Run the following Linux script to preprocess the data for a demo sequence. 
Note that this process may take some time to complete.
```bash
DATA_ROOT=<YOUR_PATH_TO_DATA>
cd radarsplat
bash boreas/data_processing/scripts/process_seq_paper.sh $DATA_ROOT boreas-2021-09-02-11-42 0.0596 68 && \
```

I you want full experiment reported in paper, run:
```bash
DATA_ROOT=<YOUR_PATH_TO_DATA>
cd radarsplat
bash boreas/data_processing/scripts/process_seq_paper.sh $DATA_ROOT boreas-2021-09-02-11-42 0.0596 411 && \
bash boreas/data_processing/scripts/process_seq_paper.sh $DATA_ROOT boreas-2021-01-26-11-22 0.0596 501 && \
bash boreas/data_processing/scripts/process_seq_paper.sh $DATA_ROOT boreas-2021-04-29-15-55 0.0596 271 && \
bash boreas/data_processing/scripts/process_seq_paper.sh $DATA_ROOT boreas-2021-09-14-20-00 0.0596 451 && \
bash boreas/data_processing/scripts/process_seq_paper.sh $DATA_ROOT boreas-2021-04-08-12-44 0.0596 386
```

The following folders will be created under each boreas sequence folder:
```bash
├── sensor.yaml
├── multipath_model
├── radar_average_map
├── radar_average_map_polar
├── radar_trajectory.tum
├── synced_lidar
└── synced_lidar_map_win5
```

## Run RadarSplat

Run demo sequence experiments and ablation studies. 
```bash
cd ~/radarsplat/examples/demo_scripts
bash run_all_radarsplat.sh ./seq_demo.txt
```

Run full experiments and ablation studies in the paper. 
```bash
cd ~/radarsplat/examples/demo_scripts
bash run_all_radarsplat.sh ./seq_all.txt
```

## Evaluation
Run demo sequence evaluation.
```bash
python eval_summary.py ./examples/demo_scripts/seq_demo.txt
```

Run full evaluation.
```bash
python eval_summary.py ./examples/demo_scripts/seq_all.txt
```

## Rendering/Visualization
Render outputs from a trained model using all available frames (train + val) for evaluation and visualization.
```bash
python examples/radar_simple_trainer.py default --ckpt <CHECKPOINT_PATH> --eval_set all --save_fig --use_lidar_map
```




## Installation

**Dependence**: Please install [Pytorch](https://pytorch.org/get-started/locally/) first.

The easiest way is to install from PyPI. In this way it will build the CUDA code **on the first run** (JIT).

```bash
pip install gsplat
```

Alternatively you can install gsplat from source. In this way it will build the CUDA code during installation.

```bash
pip install git+https://github.com/nerfstudio-project/gsplat.git
```

We also provide [pre-compiled wheels](https://docs.gsplat.studio/whl) for both linux and windows on certain python-torch-CUDA combinations (please check first which versions are supported). Note this way you would have to manually install [gsplat's dependencies](https://github.com/nerfstudio-project/gsplat/blob/6022cf45a19ee307803aaf1f19d407befad2a033/setup.py#L115). For example, to install gsplat for pytorch 2.0 and cuda 11.8 you can run
```
pip install ninja numpy jaxtyping rich
pip install gsplat --index-url https://docs.gsplat.studio/whl/pt20cu118
```

To build gsplat from source on Windows, please check [this instruction](docs/INSTALL_WIN.md).

## Evaluation

This repo comes with a standalone script that reproduces the official Gaussian Splatting with exactly the same performance on PSNR, SSIM, LPIPS, and converged number of Gaussians. Powered by gsplat’s efficient CUDA implementation, the training takes up to **4x less GPU memory** with up to **15% less time** to finish than the official implementation. Full report can be found [here](https://docs.gsplat.studio/main/tests/eval.html).

```bash
pip install -r examples/requirements.txt
# download mipnerf_360 benchmark data
python examples/datasets/download_dataset.py
# run batch evaluation
bash examples/benchmarks/basic.sh
```

## Examples

We provide a set of examples to get you started! Below you can find the details about
the examples (requires to install some exta dependencies via `pip install -r examples/requirements.txt`)

- [Train a 3D Gaussian splatting model on a COLMAP capture.](https://docs.gsplat.studio/main/examples/colmap.html)
- [Fit a 2D image with 3D Gaussians.](https://docs.gsplat.studio/main/examples/image.html)
- [Render a large scene in real-time.](https://docs.gsplat.studio/main/examples/large_scale.html)


## Development and Contribution

This repository was born from the curiosity of people on the Nerfstudio team trying to understand a new rendering technique. We welcome contributions of any kind and are open to feedback, bug-reports, and improvements to help expand the capabilities of this software.

This project is developed by the following wonderful contributors (unordered):

- [Angjoo Kanazawa](https://people.eecs.berkeley.edu/~kanazawa/) (UC Berkeley): Mentor of the project.
- [Matthew Tancik](https://www.matthewtancik.com/about-me) (Luma AI): Mentor of the project.
- [Vickie Ye](https://people.eecs.berkeley.edu/~vye/) (UC Berkeley): Project lead. v0.1 lead.
- [Matias Turkulainen](https://maturk.github.io/) (Aalto University): Core developer.
- [Ruilong Li](https://www.liruilong.cn/) (UC Berkeley): Core developer. v1.0 lead.
- [Justin Kerr](https://kerrj.github.io/) (UC Berkeley): Core developer.
- [Brent Yi](https://github.com/brentyi) (UC Berkeley): Core developer.
- [Zhuoyang Pan](https://panzhy.com/) (ShanghaiTech University): Core developer.
- [Jianbo Ye](http://www.jianboye.org/) (Amazon): Core developer.

We also have a white paper with about the project with benchmarking and mathematical supplement with conventions and derivations, available [here](https://arxiv.org/abs/2409.06765). If you find this library useful in your projects or papers, please consider citing:

```
@article{ye2024gsplatopensourcelibrarygaussian,
    title={gsplat: An Open-Source Library for {Gaussian} Splatting}, 
    author={Vickie Ye and Ruilong Li and Justin Kerr and Matias Turkulainen and Brent Yi and Zhuoyang Pan and Otto Seiskari and Jianbo Ye and Jeffrey Hu and Matthew Tancik and Angjoo Kanazawa},
    year={2024},
    eprint={2409.06765},
    journal={arXiv preprint arXiv:2409.06765},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2409.06765}, 
}
```

We welcome contributions of any kind and are open to feedback, bug-reports, and improvements to help expand the capabilities of this software. Please check [docs/DEV.md](docs/DEV.md) for more info about development.
