# DUAL

Official inference code for **DUAL: Diffusion-based Self-supervised Adaptive Denoising for Live-Cell Fluorescence Microscopy**.

DUAL is a two-stage self-supervised denoising framework for low-SNR time-lapse fluorescence microscopy. A regression network first provides a stable structural estimate, and a conditional diffusion model then restores fine details. Adaptive diffusion inversion (ADI) estimates a frame-specific starting timestep to accommodate time-varying degradation.

This repository contains the **minimal inference implementation**. Training code and pretrained model files are not included.

## Repository structure

```text
DUAL/
├── config/
│   └── demo_config.json
├── data/
│   ├── __init__.py
│   ├── dataset.py
│   └── util.py
├── demo_data/
│   └── test.tif
├── model/
│   ├── __init__.py
│   ├── adaptive_inversion.py
│   ├── base_model.py
│   ├── diffusion.py
│   ├── model_stage1.py
│   ├── model_stage2.py
│   ├── network_stage1.py
│   ├── network_stage2.py
│   ├── unet.py
│   └── utils.py
├── pretrained/
│   ├── stage1/
│   │   └── I22000_E0067_gen.pth
│   └── stage2/
│       └── I50000_E151_gen.pth
├── utils/
│   ├── logger.py
│   └── metrics.py
├── requirements.txt
└── train_s2.py
```

## Installation
We recommend using Git LFS to download the code. The download command is as follows:
```bash
git clone git@github.com:feilab-hust/DUAL.git
```
Python 3.9 is recommended.

```bash
conda create -n dual python=3.9
conda activate dual
pip install -r requirements.txt
```

A CUDA-capable NVIDIA GPU is required for the default configuration.

## Pretrained models

The pretrained weights are not stored in this repository. Place them at the following exact paths:

```text
pretrained/stage1/I22000_E0067_gen.pth
pretrained/stage2/I50000_E151_gen.pth
```

The paths in `config/demo_config.json` intentionally omit the `_gen.pth` suffix because the code appends it automatically.

## Input data

The default example is:

```text
demo_data/test.tif
```

The provided TIFF is interpreted as a time-lapse stack with shape:

```text
T × H × W
```

Internally, the loader organizes data as:

```text
N × T × Z × H × W
```

For a 3D TIFF with `time_seq: true`, the first dimension is treated as time and the data are represented as `1 × T × 1 × H × W`.

The current demo configuration uses:

- adjacent temporal frames as local conditions;
- the first frame as the global reference condition;
- no explicit inter-slice spatial condition;
- slice-wise min-max normalization.

## Run inference

From the repository root, run:

```bash
python train_s2.py
```

The default arguments are equivalent to:

```bash
python train_s2.py \
  -c config/demo_config.json \
  -p val \
  -gpu 0 \
  -name dual_demo
```

To select another GPU, for example GPU 1:

```bash
python train_s2.py -gpu 1
```

## Adaptive diffusion inversion

At the first run, DUAL computes the matched diffusion timestep for each input frame and creates:

```text
experiments/dual_demo_s2/adaptive_matching.txt
```

If this file already exists and is non-empty, it is reused. Delete it before rerunning inference on a different input dataset under the same experiment name.

## Output

The default denoised result is written to:

```text
experiments/dual_demo_s2/results/validation_test.tif/
```

The directory contains:

```text
S2_I50000_E151_denoised.tif
input.tif
```

## Use another TIFF stack

Edit the validation data path in `config/demo_config.json`:

```json
"datasets": {
  "val": {
    "dataroot": "path/to/your_data.tif"
  }
}
```

Keep the network architecture and checkpoint prefixes unchanged when using the provided pretrained models. When changing the input dataset, also remove the previous `adaptive_matching.txt` or use a new experiment name.

## Citation

```bibtex
@article{zhang2026dual,
  title   = {Diffusion-based Self-supervised Adaptive Denoising for Live-Cell Fluorescence Microscopy},
  author  = {Zhang, Zihao and Sun, Minglu and Yi, Chengqiang and Mao, Shiqi and Liu, Ying and Zhou, Yao and Liu, Binbing and Fei, Peng},
  year    = {2026}
}
```

## License

A license file has not yet been added. Please contact the authors before redistributing or using the code for commercial purposes.
