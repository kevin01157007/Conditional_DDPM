# Conditional DDPM for Multi-Label Image Generation (i-CLEVR)

This repository implements a **Conditional Denoising Diffusion Probabilistic Model (Conditional DDPM)** for multi-label image generation on the **i-CLEVR** dataset (Deep Learning Lab 6, Fall 2026, NYCU). Given a set of 1 to 3 object descriptions (8 colors $\times$ 3 shapes = 24 categories), the model synthesizes $64 \times 64$ images containing the specified objects.

---

## Project Structure

```text
Conditional_DDPM/
├── spec/
│   └── Lab6_Conditional_DDPM_2026_Fall.pdf  # Lab specification
├── data/
│   └── iclevr/                              # Training images (download separately)
│       ├── CLEVR_train_000000_0.png
│       └── ...
├── checkpoints/                             # Saved model weights
│   └── best_model.pth
├── images/                                  # Generated outputs
│   ├── test/                                # test.json PNGs (0.png ~ 31.png)
│   ├── new_test/                            # new_test.json PNGs (0.png ~ 31.png)
│   ├── grid_test.png                        # 8x4 grid for test.json
│   ├── grid_new_test.png                    # 8x4 grid for new_test.json
│   └── denoising_process.png                # Denoising steps grid
├── model/
│   ├── unet.py                              # Conditional U-Net & label embedding
│   └── diffusion.py                         # Noise schedule, CFG & sampling logic
├── src/
│   ├── dataset.py                           # i-CLEVR Dataset & DataLoader
│   ├── train.py                             # Training script (CFG & EMA)
│   └── sample.py                            # Inference, evaluation & visualization
├── evaluator.py                             # Pretrained ResNet-18 evaluator
├── checkpoint.pth                           # Evaluator weights
├── objects.json                             # 24 object label-to-index map
├── train.json                               # Training metadata (18,009 samples)
├── test.json                                # Test set 1 (32 conditions)
├── new_test.json                            # Test set 2 (32 conditions)
├── readme.txt                               # Original dataset description
└── README.md                                # Project documentation
```

---

## Quick Start

### 1. Environment Setup

Install the required dependencies (PyTorch with CUDA support, `torchvision`, `diffusers`, `accelerate`, `tqdm`, `Pillow`):

```bash
pip install torch torchvision diffusers accelerate tqdm pillow
```

### 2. Dataset Preparation

1. Download the `iclevr` training images from the course Google Drive link provided in the specification.
2. Extract the image files into `d:/Conditional_DDPM/iclevr/` so that paths like `iclevr/CLEVR_train_001032_0.png` match the keys in `train.json`.
3. Ensure `checkpoint.pth` (the evaluator weight file) is in the root directory `./checkpoint.pth`.

### 3. Training

Run `train.py` to train the Conditional DDPM with Classifier-Free Guidance (CFG):

```bash
python src/train.py \
    --data_dir ./data/iclevr \
    --train_json ./train.json \
    --objects_json ./objects.json \
    --output_dir ./checkpoints \
    --epochs 50 \
    --batch_size 64 \
    --lr 1e-4 \
    --guidance_scale 3.5
```

* Model checkpoints and periodic validation accuracy logs will be saved to `./checkpoints/`.

### 4. Inference, Evaluation & Visualization

Run `sample.py` to generate the required images for `test.json` and `new_test.json`, compute Top-k accuracies using `evaluator.py`, and export all required grids:

```bash
python src/sample.py \
    --ckpt ./checkpoints/best_model.pth \
    --test_json ./test.json \
    --new_test_json ./new_test.json \
    --objects_json ./objects.json \
    --output_dir ./images \
    --guidance_scale 3.5 \
    --num_inference_steps 100
```

Running this script will automatically produce:
- `images/test/0.png` ... `31.png` and `images/new_test/0.png` ... `31.png`
- `images/grid_test.png` and `images/grid_new_test.png` ($8 \times 4$ grids)
- `images/denoising_process.png` (at least 8 denoising steps for `{"blue cube", "yellow sphere", "purple cylinder"}`)
- Console output of the evaluator accuracy for both `test.json` and `new_test.json`
