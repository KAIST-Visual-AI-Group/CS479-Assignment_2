<div align=center>
  <h1>
    NeRF: 3D Reconstruction from 2D Images
  </h1>
  <p>
    <a href=https://mhsung.github.io/kaist-cs479-fall-2023/ target="_blank"><b>KAIST CS479: Machine Learning for 3D Data (Fall 2023)</b></a><br>
    Programming Assignment 2    
  </p>
</div>

<div align=center>
  <p>
    Instructor: <a href=https://mhsung.github.io target="_blank"><b>Minhyuk Sung</b></a> (mhsung [at] kaist.ac.kr)<br>
    TA: <a href=https://dvelopery0115.github.io target="_blank"><b>Seungwoo Yoo</b></a>  (dreamy1534 [at] kaist.ac.kr)      
  </p>
</div>

<div align=center>
  <img src="./media/nerf_blender/lego.gif" width="400" />
</div>

#### Due: October 9, 2023 (Monday) 23:59 KST
#### Where to Submit: Gradescope

## Abstract

The introduction of [Neural Radiance Fields (NeRF)](https://arxiv.org/abs/2003.08934) was a massive milestone in image-based, neural rendering literature.
Compared with previous works on novel view synthesis, NeRF is a simple, yet powerful idea that combines recently emerging neural implicit representations with traditional volume rendering techniques.
As of today, the follow-up research aiming to scale and extend the idea to various tasks has become one of the most significant streams in the computer vision community thanks to its simplicity and versatility.

In this assignment, we will take a technical deep dive into NeRF to understand this ground-breaking approach which will help us navigate a broader landscape of the field.
We strongly recommend you check out the paper, together with [our brief summary](https://geometry-kaist.notion.site/Tutorial-2-NeRF-Neural-Radiance-Field-ef0c1f3446434162a540e6afc7aeccc8?pvs=4), before, or while working on this assignment.

> :warning: **This assignment involves training a neural network that takes approximately 2 hours. Start as early as possible.**

<details>
<summary><b>Table of Content</b></summary>
  
- [Abstract](#abstract)
- [Setup](#setup)
- [Code Structure](#code-structure)
- [Tasks](#tasks)
  - [Task 0. Download Data](#task-0-download-data)
  - [Task 1. Implementing MLP](#task-1-implementing-mlp)
  - [Task 2. Implementing Ray Sampling](#task-2-implementing-ray-sampling)
  - [Task 3. Implementing Volume Rendering Equation](#task-3-implementing-volume-rendering-equation)
  - [Task 4. Qualitative \& Quantitative Evaluation](#task-4-qualitative--quantitative-evaluation)
  - [(Optional) Task 5. Train NeRF with Your Own Data](#optional-task-5-train-nerf-with-your-own-data)
- [What to Submit](#what-to-submit)
- [Grading](#grading)
- [Further Readings](#further-readings)
</details>

## Setup

We recommend creating a virtual environment using `conda`.
To create a `conda` environment, issue the following command:
```
conda create --name nerf-tutorial python=3.8
```
This should create a basic environment with Python 3.8 installed.
Next, activate the environment and install the dependencies using `pip`:
```
conda activate nerf-tutorial
pip install -r requirements.txt
```
The remaining dependencies are the ones related to PyTorch and they can be installed with the command:
```
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install torchmetrics[image]
pip install tensorboard
```

Register the project root directory (i.e., `torch-NeRF`) as an environment variable to help the Python interpreter search our files.
```
export PYTHONPATH=.
```

By default, the configuration is set for the `lego` scene included in the `Blender` dataset. Refer to the config files under `config` for more details. Executing the following initiates training:
```
python torch_nerf/runners/train.py
```
All by-products produced during each run, including TensorBoard logs, will be saved under an experiment directory under `outputs`. This is automatically done by [Hydra](https://hydra.cc), the library we use for managing our config files. Refer to [the official documentation](https://hydra.cc/docs/intro/) for examples and APIs.

**We highly encourage you to try out multiple seeds as the performance of neural networks is often sensitive to the initialization.**
The function `init_torch` that sets the random seed for PyTorch is located at `torch_nerf/runners/utils.py`

> :bulb: **Each run takes approximately 2 hours on a single NVIDIA RTX 3090 GPU and consumes around 10 GB of VRAM.**

After training NeRF, it can be rendered using the script `render.py.`
To do so, provide the experiment directory created when running the training script. For instance,
```
python torch_nerf/runners/render.py +log_dir=outputs/2023-06-27/00-10-15 +render_test_views=False
```
The Boolean flag `render_test_views` determines whether to render the trained scene from the viewpoints held out for testing. We will come back to this when discussing quantitative evaluation.

## Code Structure
This codebase is organized as the following directory tree. We only list the core components for brevity:
```
torch_nerf
│
├── configs             <- Directory containing config files
│
├── runners
│   ├── evaluate.py     <- Script for quantitative evaluation.
│   ├── render.py       <- Script for rendering (i.e., qualitative evaluation).
│   ├── train.py        <- Script for training.
│   └── utils.py        <- A collection of utilities used in the scripts above.
│
├── src
│   ├── cameras
│   │   ├── cameras.py
│   │   └── rays.py
│   │   
│   ├── network
│   │   └── nerf.py
│   │
│   ├── renderer
│   │   ├── integrators
│   │   ├── ray_samplers
│   │   └── volume_renderer.py
│   │
│   ├── scene
│   │
│   ├── signal_encoder
│   │   ├── positional_encoder.py
│   │   └── signal_encoder_base.py
│   │
│   └── utils
│       ├── data
│       │   ├── blender_dataset.py
│       │   └── load_blender.py
│       │
│       └── metrics
│           └── rgb_metrics.py
│
├── requirements.txt    <- Dependency configuration file.
└── README.md           <- This file.
```

## Tasks

### Task 0. Download Data

Download the file `lego.zip` from [here](https://drive.google.com/file/d/1EitqzKZLptJop82hdNqu1YCogxgNgN5u/view?usp=share_link) and extract it under directory `data/nerf_synthetic`. The training script expects the data to be located under `data/nerf_synthetic/lego`.

> :bulb: `scp` is a handy tool for transferring files between local and remote servers. Check [this link](https://haydenjames.io/linux-securely-copy-files-using-scp/) for examples.

### Task 1. Implementing MLP
<p align="middle">
  <img src="./media/nerf_mlp.png" width="800" />
</p>

```bash
#! files-to-modify
$ torch_nerf/src/network/nerf.py
```
Implement the MLP displayed above. The network consists of:

1. One input fully-connected layer;
2. Nine fully-connected layers (including the one for skip connection);
3. One output fully-connected layer.

All hidden layers are followed by ReLU activation, and the density and the RGB head at the output layer are followed by ReLU and sigmoid activations, respectively.
For more details, please refer to Sec. A of the paper's supplementary material.

> :bulb: We highly recommend you to look up [the official documentation](https://pytorch.org/docs/stable/nn.html) of the layers used in the network.

### Task 2. Implementing Ray Sampling
```bash
#! files-to-modify
$ torch_nerf/src/cameras/rays.py
$ torch_nerf/src/renderer/ray_samplers/stratified_sampler.py
```
This task consists of two sub-tasks:

1. Implement the body of function `compute_sample_coordinates` in `torch_nerf/src/cameras/rays.py`.
This function will be used to evaluate the coordinates of points along rays cast from image pixels.
For a ray $r$ parameterized by the origin $\mathbf{o}$ and direction $\mathbf{d}$ (not necessarily a unit vector), a point on the ray can be computed by

```math
r(t) = \mathbf{o} + t \mathbf{d},
```
where $t \in [t_n, t_f]$ is bounded by the near bound $t_n$ and the far bound $t_f$, respectively.

2. Implement the body of function `sample_along_rays_uniform` in `torch_nerf/src/renderer/ray_samplers/stratified_sampler.py`.
The function implements the stratified sampling illustrated in the following equation (Eqn 2. in the paper).

```math
t_i \sim \mathcal{U} \left[ t_n + \frac{i-1}{N} \left( t_f - t_n \right), t_n + \frac{i}{N} \left( t_f - t_n \right) \right].
```

> :bulb: Check out the helper functions [`create_t_bins`](https://github.com/KAIST-Geometric-AI-Group/CS479-Assignment-2/blob/main/torch_nerf/src/renderer/ray_samplers/stratified_sampler.py#L110) and [`map_t_to_euclidean`](https://github.com/KAIST-Geometric-AI-Group/CS479-Assignment-2/blob/main/torch_nerf/src/renderer/ray_samplers/stratified_sampler.py#L97) while implementing function `sample_along_rays_uniform`. Also, you may find [`torch.rand_like`](https://pytorch.org/docs/stable/generated/torch.rand_like.html) useful when generating random numbers for sampling.

> :bulb: Note that all rays in a ray bundle share the same near and far bounds. Although function `map_t_to_euclidean` takes only `float` as its arguments `near` and `far`, it is not necessary to loop over all rays individually.

### Task 3. Implementing Volume Rendering Equation
```bash
#! files-to-modify
$ torch_nerf/src/renderer/integrators/quadrature_integrator.py
```
This task consists of one sub-task:

1. Implement the body of function `integrate_along_rays`.
The function implements Eqn. 3 in the paper which defines a pixel color as a weighted sum of radiance values collected along a ray:

```math
\hat{C} \left( r \right) = \sum_{i=1}^T T_i \left( 1 - \exp \left( -\sigma_i \delta_i \right) \right) \mathbf{c}_i,
```
where 
```math
T_i = \exp \left( - \sum_{j=1}^{i-1} \sigma_j \delta_j \right).
```

> :bulb: The PyTorch APIs [`torch.exp`](https://pytorch.org/docs/stable/generated/torch.exp.html?highlight=exp#torch.exp), [`torch.cumsum`](https://pytorch.org/docs/stable/generated/torch.cumsum.html?highlight=cumsum#torch.cumsum), and [`torch.sum`](https://pytorch.org/docs/stable/generated/torch.sum.html?highlight=sum#torch.sum) might be useful when implementing the quadrature integration.

### Task 4. Qualitative \& Quantitative Evaluation

For qualitative evaluation, render the trained scene with the provided script. 
```
python torch_nerf/runners/render.py +log_dir=${LOG_DIR} +render_test_views=False
```
This will produce a set of images rendered while orbiting around the upper hemisphere of an object.
The rendered images can be compiled into a video using the script `scripts/utils/create_video.py`:
```
python scripts/utils/create_video.py --img_dir ${RENDERED_IMG_DIR} --vid_title ${VIDEO_TITLE}
```

For quantitative evaluation, render the trained scene again, **but from the test views**.
```
python torch_nerf/runners/render.py +log_dir=${LOG_DIR} +render_test_views=True
```
This will produce 200 images (in the case of the synthetic dataset) held out during training.
After rendering images from the test view, use the script `evaluate.py` to compute LPIPS, PSNR, and SSIM. For instance, to evaluate the implementation for the `lego` scene:
```
python torch_nerf/runners/evaluate.py ${RENDERED_IMG_DIR} ./data/nerf_synthetic/lego/test
```
The metrics measured after training the network for 50k iterations on the `lego` scene are summarized in the following table.
| LPIPS (↓) | PSNR (↑) | SSIM (↑) |
|---|---|---|
| 0.0481 | 28.9258 | 0.9473 |

> :bulb: **For details on grading, refer to section [Evaluation Criteria](#evaluation-criteria).**

### (Optional) Task 5. Train NeRF with Your Own Data

Instead of using the provided dataset, capture your surrounding environment and use the data for training.
[COLMAP](https://github.com/colmap/colmap) might be useful when computing the relative camera poses.

## What to Submit

Compile the following files as a **ZIP** file named `{NAME}_{STUDENT_ID}.zip` and submit the file via Gradescope.
  
- The folder `torch_nerf` that contains every source code file;
- A folder named `{NAME}_{STUDENT_ID}_renderings` containing the renderings (`.png` files) from the **test views** used for computing evaluation metrics;
- A text file named `{NAME}_{STUDENT_ID}.txt` containing **a comma-separated list of LPIPS, PSNR, and SSIM** from quantitative evaluation;
- The checkpoint file named `{NAME}_{STUDENT_ID}.pth` used to produce the above metrics.

## Grading

**You will receive a zero score if:**
- **you do not submit,**
- **your code is not executable in the Python environment we provided, or**
- **you modify any code outside of the section marked with `TODO`.**
  
**Plagiarism in any form will also result in a zero score and will be reported to the university.**

**Your score will incur a 10% deduction for each missing item in the [Submission Guidelines](#submission-guidelines) section.**

Otherwise, you will receive up to 300 points from this assignment that count toward your final grade.

| Evaluation Criterion | LPIPS (↓) | PSNR (↑) | SSIM (↑) |
|---|---|---|---|
| **Success Condition \(100%\)** | **0.06** | **28.00** | **0.90** |
| **Success Condition \(50%)**   | **0.10**  | **20.00** | **0.60** |

As shown in the table above, each evaluation metric is assigned up to 100 points. In particular,
- **LPIPS**
  - You will receive 100 points if the reported value is equal to or, *smaller* than the success condition \(100%)\;
  - Otherwise, you will receive 50 points if the reported value is equal to or, *smaller* than the success condition \(50%)\.
- **PSNR**
  - You will receive 100 points if the reported value is equal to or, *greater* than the success condition \(100%)\;
  - Otherwise, you will receive 50 points if the reported value is equal to or, *greater* than the success condition \(50%)\.
- **SSIM**
  - You will receive 100 points if the reported value is equal to or, *greater* than the success condition \(100%)\;
  - Otherwise, you will receive 50 points if the reported value is equal to or, *greater* than the success condition \(50%)\.

## Further Readings

If you are interested in this topic, we encourage you to check out the papers listed below.

- [NeRF++: Analyzing and Improving Neural Radiance Fields (arXiv 2021)](https://arxiv.org/abs/2010.07492)
- [NeRF in the Wild: Neural Radiance Fields for Unconstrained Photo Collections (CVPR 2021)](https://arxiv.org/abs/2008.02268)
- [pixelNeRF: Neural Radiance Fields from One or Few Images (CVPR 2021)](https://arxiv.org/abs/2012.02190)
- [Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields (ICCV 2021)](https://arxiv.org/abs/2103.13415)
- [BARF: Bundle-Adjusting Neural Radiance Fields (ICCV 2021)](https://arxiv.org/abs/2104.06405)
- [Nerfies: Deformable Neural Radiance Fields (ICCV 2021)](https://arxiv.org/abs/2011.12948)
- [NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction (NeurIPS 2021)](https://arxiv.org/abs/2106.10689)
- [Volume Rendering of Neural Implicit Surfaces (NeurIPS 2021)](https://arxiv.org/abs/2106.12052)
- [Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields (CVPR 2022)](https://arxiv.org/abs/2111.12077)
- [RegNeRF: Regularizing Neural Radiance Fields for View Synthesis from Sparse Inputs (CVPR 2022)](https://arxiv.org/abs/2112.00724)
- [Mega-NeRF: Scalable Construction of Large-Scale NeRFs for Virtual Fly-Throughs (CVPR 2022)](https://arxiv.org/abs/2112.10703)
- [Plenoxels: Radiance Fields without Neural Networks (CVPR 2022)](https://arxiv.org/abs/2112.05131)
- [Point-NeRF: Point-based Neural Radiance Fields (CVPR 2022)](https://arxiv.org/abs/2201.08845)
- [Instant-NGP: Instant Neural Graphics Primitives with a Multiresolution Hash Encoding (SIGGRAPH 2022)](https://arxiv.org/abs/2201.05989)
- [TensoRF: Tensorial Radiance Fields (ECCV 2022)](https://arxiv.org/abs/2203.09517)
- [MobileNeRF: Exploiting the Polygon Rasterization Pipeline for Efficient Neural Field Rendering on Mobile Architectures (CVPR 2023)](https://arxiv.org/abs/2208.00277v5)
- [Zip-NeRF: Anti-Aliased Grid-Based Neural Radiance Fields (ICCV 2023)](https://arxiv.org/abs/2304.06706)
