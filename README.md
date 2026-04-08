<div align="center">
<h1>TexGS-VolVis:<br>Expressive Scene Editing for Volume Visualization via Textured Gaussian Splatting</h1>

<a href="https://doi.org/10.1109/TVCG.2025.3634643" target="_blank">
    <img alt="Paper" src="https://img.shields.io/badge/Paper-TVCG-red?logo=ieee" height="20" /></a>
<a href="https://github.com/TouKaienn/TexGS-VolVis" target="_blank">
    <img alt="Code" src="https://img.shields.io/badge/Code-TexGS--VolVis-blue?logo=github" height="20" /></a>

<div>
    <a href="https://toukaienn.github.io/" target="_blank">Kaiyuan Tang</a><sup>1</sup>,
    <a href="https://kuangshiai.github.io/" target="_blank">Kuangshi Ai</a><sup>1</sup>,
    <a href="https://stevenhan1991.github.io/" target="_blank">Jun Han</a><sup>2</sup>,
    <a href="https://sites.nd.edu/chaoli-wang/" target="_blank">Chaoli Wang</a><sup>1</sup>
</div>

<div>
    <sup>1</sup>University of Notre Dame&emsp;
    <sup>2</sup>Hong Kong University of Science and Technology
</div>

</div>

<br>

![Workflow](assets/workflow.png)

TexGS-VolVis is a textured Gaussian splatting framework for volume visualization. It extends 2D Gaussian primitives with learnable texture and shading attributes, enabling geometry-consistent stylization, enhanced lighting control, and real-time rendering. The framework supports image- and text-driven non-photorealistic scene editing, palette-based recoloring, relighting, and 2D-lift-3D segmentation for partial editing with fine-grained control.


## Installation

### 1. Create Conda Environment

```bash
conda create -n texgs python=3.10 -y
conda activate texgs
```

### 2. Install PyTorch (CUDA 11.8)

```bash
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Compile CUDA Submodules

All five submodules must be compiled. Run from the project root:

```bash
pip install submodules/simple-knn
pip install submodules/diff-surfel-rasterization
pip install submodules/2DGSTex
pip install submodules/2DGSTexLighting
pip install submodules/compute_normal_cuda
```

> **Note**: If compilation fails, ensure `nvcc` is on your PATH and `CUDA_HOME` is set:
> ```bash
> export CUDA_HOME=/usr/local/cuda
> export PATH=$CUDA_HOME/bin:$PATH
> ```

### 5. (Optional) Pre-download Models for Text Editing

Text-guided editing requires Hugging Face models. They are auto-downloaded on first run, or you can pre-download:

```bash
# InstructPix2Pix (for textEdit.py)
python -c "from diffusers import StableDiffusionInstructPix2PixPipeline; StableDiffusionInstructPix2PixPipeline.from_pretrained('timbrooks/instruct-pix2pix')"
```

## Data Preparation

The system expects data in Blender/NeRF synthetic format. Each transfer function (TF) is a separate subdirectory containing multi-view rendered images and camera transforms:

```
<dataset_root>/                         # e.g., StaticImgData/noPC/vortex
├── TF01/
│   ├── train/                          # Training images (r_0000.png, r_0001.png, ...)
│   ├── test/                           # Test images
│   ├── transforms_train.json           # Camera transforms for training views
│   ├── transforms_test.json            # Camera transforms for test views
│   ├── transforms_val.json             # Camera transforms for validation views
│   └── points3d.ply                    # (Optional) Initial point cloud
├── TF02/
│   └── ...
├── TF03/
│   └── ...
└── TF04/
    └── ...
```

A sample dataset (`vortex`) is provided in the `Data/` folder for reference.

## Usage

### Two-Stage Training Pipeline

The standard workflow trains a base 2DGS model first, then fine-tunes it into a TexGS model with learnable textures.

#### Stage 1: Train 2DGS Base Model

```bash
python train.py --eval \
    -s <data_path>/TF01 \
    -m ./output/<exp_name>/TF01/2dgs
```

Key arguments:
- `-s` / `--source_path`: Path to a TF dataset (with `train/`, `test/`, and `transforms_*.json`)
- `-m` / `--model_path`: Output model directory
- `--eval`: Enable train/test split for evaluation
- `--iteration`: Training iterations (default: 30000)
- `--white_background`: Use white background (default: True)

#### Stage 2: Train TexGS with Learnable Textures

```bash
python train.py --eval \
    -t TexGS \
    -s <data_path>/TF01 \
    -m ./output/<exp_name>/TF01/texgs \
    -init ./output/<exp_name>/TF01/2dgs/point_cloud/iteration_30000/point_cloud.ply \
    --iteration 3000
```

Key arguments:
- `-t TexGS`: Set training type to TexGS (options: `2DGS`, `TexGS`, `stylize`)
- `-init`: Path to the 2DGS checkpoint (.ply) for initialization

#### Automated Training (All TFs)

```bash
bash scripts/run.sh <dataset_root> <exp_name>
```

This trains both 2DGS and TexGS for every `TF*` subdirectory in `<dataset_root>`.

### Rendering / Inference

Render trained TexGS models and composite multiple TFs:

```bash
python render.py \
    -so ./output/<exp_name> \
    --source_path <data_path>/TF01 \
    --output_dir ./output/render_output_<exp_name>
```

Key arguments:
- `-so` / `--source_dir`: Directory containing trained TF models
- `--output_dir`: Where to save rendered images
- `--style_names`: Specify which style checkpoint to use per TF (default: `texgs`)
- `--stylized_texture`: Use stylized textures (sets palette to black)
- `--skip_mesh`: Skip mesh extraction (default: True)

Batch rendering:

```bash
bash scripts/render.sh <dataset_root> <exp_name>
```

### Text-Guided Texture Editing

Edit textures using natural language prompts via InstructPix2Pix:

```bash
python textEdit.py --eval \
    -t stylize \
    -s <data_path>/TF01 \
    -m ./output/<exp_name>/TF01/<edit_name> \
    -init ./output/<exp_name>/TF01/2dgs/point_cloud/iteration_30000/point_cloud.ply \
    --iteration 1500 \
    --text_prompt "Make it look like marble"
```

Or use the convenience script:

```bash
bash scripts/edit.sh <dataset_root> <exp_name> "Make it look like marble" <edit_name>
```

### Image-Guided Style Transfer

Transfer visual styles from a reference image:

```bash
python imgEdit.py --eval \
    -t stylize \
    -s <data_path>/TF01 \
    -m ./output/<exp_name>/TF01/<edit_name>_Img \
    -init ./output/<exp_name>/TF01/2dgs/point_cloud/iteration_30000/point_cloud.ply \
    --iteration 3000 \
    --style_img_path <path_to_style_image>
```

Or batch:

```bash
bash scripts/img_edit.sh <dataset_root> <exp_name> <edit_name> <style_img_1> <style_img_2> ...
```

### 3D Texture Painting

Interactive painting on 3D surfaces:

```bash
python paint3d.py --eval \
    -t stylize \
    -s <data_path>/TF01 \
    -m ./output/<exp_name>/TF01/<edit_name> \
    -init ./output/<exp_name>/TF01/2dgs/point_cloud/iteration_30000/point_cloud.ply \
    --iteration 4000 \
    --text_prompt "Add blue highlights"
```

### Interactive GUI

Launch the interactive viewer and editor:

```bash
python gui.py \
    --source_path <data_path>/TF01 \
    --model_path ./output/<exp_name>/TF01/texgs
```

GUI features:
- Real-time rendering with adjustable camera
- Palette-based recoloring
- Segmentation (SAM-based) for selective editing
- Lighting direction and intensity control
- Multi-TF scene composition

### Relighting

Render with different lighting conditions:

```bash
python render_relighting.py \
    -m ./output/<exp_name>/TF01/texgs \
    -s <data_path>/TF01
```

### Evaluation Metrics

Compute PSNR, SSIM, and LPIPS:

```bash
python metrics.py -m ./output/<exp_name>/TF01/texgs
```

## Render Modes

The system supports five rendering backends configured via the `-t` flag:

| Flag | Renderer | Description |
|------|----------|-------------|
| `2DGS` | `disk_render` | Base 2DGS surfel rendering |
| `TexGS` | `texGS_render_wLight` | TexGS with learnable lighting |
| `TexGS_noLight` | `texGS_render_woLight` | TexGS without lighting effects |
| `stylize` | `stylize_render` | Style transfer training renderer |
| `stylize_inf` | `stylize_render_inf` | Style transfer inference renderer |

## Key Configuration Parameters

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--iteration` | 30000 (2DGS) / 3000 (TexGS) | Training iterations |
| `--sh_degree` | 3 | Spherical harmonics degree |
| `--white_background` | True | Background color |
| `--texture_lr` | 0.0025 | Texture learning rate |
| `--palette_color_lr` | 0.01 | Palette color learning rate |
| `--pixel_num` | 1e7 | Total texture resolution |
| `--build_chart_every` | 100 | Texture chart rebuild frequency |
| `--densify_from_iter` | 500 | Start Gaussian densification |
| `--densify_until_iter` | 13500 | Stop Gaussian densification |

### Editing Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--text_prompt` | - | Natural language editing instruction |
| `--style_img_path` | - | Reference style image path |
| `--edit_name` | - | Output folder name for edits |
| `--edit_steps` | 50 | Edit every N training steps |

## Citation

If you find this work useful, please cite:

```bibtex
@article{TexGS-VolVis,
  author={Tang, Kaiyuan and Ai, Kuangshi and Han, Jun and Wang, Chaoli},
  journal={IEEE Transactions on Visualization and Computer Graphics}, 
  title={TexGS-VolVis: Expressive Scene Editing for Volume Visualization via Textured Gaussian Splatting}, 
  year={2026},
  volume={32},
  number={1},
  pages={933-943},
  doi={10.1109/TVCG.2025.3634643}
  }
```

## Acknowledgements

- [2D Gaussian Splatting (2DGS)](https://surfel-gs.github.io/) - Huang et al.
- [GStex](https://github.com/victor-rong/GStex) - Victor Rong et al.
- [InstructPix2Pix](https://www.timothybrooks.com/instruct-pix2pix) - Brooks et al.
- [Segment Anything (SAM)](https://segment-anything.com/) - Kirillov et al.
