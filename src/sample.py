import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model.diffusion import ConditionalDDPM
from model.unet import ConditionalUNet
from src.dataset import encode_labels, load_condition_file, load_object_map

# Lab 6 規格指定的去噪過程圖條件
DENOISING_CONDITION = ["blue cube", "yellow sphere", "purple cylinder"]


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """將 [-1, 1] 張量還原為 [0, 1] 供存圖與視覺化使用。"""
    return (x.clamp(-1.0, 1.0) + 1.0) / 2.0


def save_generated_outputs(
    samples: torch.Tensor,
    indiv_dir: Path,
    grid_path: Path,
    nrow: int = 8,
) -> None:
    """儲存個別 PNG（0.png ~ 31.png）與 8x4 總覽格狀圖。"""
    indiv_dir.mkdir(parents=True, exist_ok=True)
    grid_path.parent.mkdir(parents=True, exist_ok=True)

    imgs_01 = denormalize(samples.cpu())
    for idx, img in enumerate(imgs_01):
        save_image(img, indiv_dir / f"{idx}.png")

    grid = make_grid(imgs_01, nrow=nrow)
    save_image(grid, grid_path)


def load_model_and_diffusion(
    ckpt_path: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[ConditionalUNet, ConditionalDDPM, int]:
    """讀取 checkpoint，建立 ConditionalUNet 與對應設定的 ConditionalDDPM。"""
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}

    num_classes = config.get("num_classes", 24)
    sample_size = config.get("sample_size", args.image_size)
    block_out_channels = tuple(config.get("block_out_channels", (128, 128, 256, 512)))

    model = ConditionalUNet(
        num_classes=num_classes,
        sample_size=sample_size,
        block_out_channels=block_out_channels,
    ).to(device)

    if isinstance(ckpt, dict) and "ema_state_dict" in ckpt and not args.use_raw_weights:
        model.load_state_dict(ckpt["ema_state_dict"])
        weight_source = "ema_state_dict"
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        weight_source = "model_state_dict"
    else:
        model.load_state_dict(ckpt)
        weight_source = "raw state_dict"

    model.eval()
    print(f"Loaded checkpoint '{ckpt_path}' using {weight_source}.")

    diffusion = ConditionalDDPM(
        num_train_timesteps=config.get("num_train_timesteps", args.num_train_timesteps),
        beta_schedule=config.get("beta_schedule", args.beta_schedule),
        p_uncond=config.get("p_uncond", 0.1),
        prediction_type=config.get("prediction_type", args.prediction_type),
    )
    return model, diffusion, sample_size


def prepare_for_evaluator(samples: torch.Tensor) -> torch.Tensor:
    """確保送進 evaluator.py 的圖片為 (B, 3, 64, 64) 且維持 [-1, 1] 範圍。"""
    images = samples.clamp(-1.0, 1.0)
    if images.shape[-2:] != (64, 64):
        images = F.interpolate(images, size=(64, 64), mode="bilinear", align_corners=False)
    return images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample, evaluate, and visualize Conditional DDPM outputs")
    parser.add_argument("--ckpt", type=str, default="./checkpoints/best_model.pth")
    parser.add_argument("--test_json", type=str, default="./test.json")
    parser.add_argument("--new_test_json", type=str, default="./new_test.json")
    parser.add_argument("--objects_json", type=str, default="./objects.json")
    parser.add_argument("--output_dir", type=str, default="./images")

    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--num_inference_steps", type=int, default=100)
    parser.add_argument("--sampler", type=str, default="ddim", choices=["ddim", "ddpm"])
    parser.add_argument("--eta", type=float, default=0.0, help="僅 DDIM 有效，0.0 為確定性取樣")
    parser.add_argument("--num_trajectory_steps", type=int, default=8, help="去噪過程圖步數（至少 8 步）")

    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--beta_schedule", type=str, default="squaredcos_cap_v2")
    parser.add_argument("--prediction_type", type=str, default="epsilon")

    parser.add_argument("--use_raw_weights", action="store_true", help="改用非 EMA 權重推論（做消融實驗用）")
    parser.add_argument("--skip_eval", action="store_true", help="略過 ResNet-18 evaluator 評分")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 載入模型與擴散排程
    model, diffusion, image_size = load_model_and_diffusion(Path(args.ckpt), args, device)

    # 2. 載入 test.json 與 new_test.json 條件
    test_labels = load_condition_file(args.test_json, args.objects_json)
    new_test_labels = load_condition_file(args.new_test_json, args.objects_json)

    # 3. 生成 test.json 圖片與 8x4 grid
    gen_test = torch.Generator(device="cpu").manual_seed(args.seed)
    test_samples = diffusion.sample(
        model=model,
        y=test_labels.to(device),
        image_size=image_size,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        sampler=args.sampler,
        eta=args.eta,
        generator=gen_test,
        show_progress=True,
    )
    save_generated_outputs(
        test_samples,
        indiv_dir=output_dir / "test",
        grid_path=output_dir / "grid_test.png",
        nrow=8,
    )

    # 4. 生成 new_test.json 圖片與 8x4 grid
    gen_new_test = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    new_test_samples = diffusion.sample(
        model=model,
        y=new_test_labels.to(device),
        image_size=image_size,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        sampler=args.sampler,
        eta=args.eta,
        generator=gen_new_test,
        show_progress=True,
    )
    save_generated_outputs(
        new_test_samples,
        indiv_dir=output_dir / "new_test",
        grid_path=output_dir / "grid_new_test.png",
        nrow=8,
    )

    # 5. 生成指定條件 {blue cube, yellow sphere, purple cylinder} 的去噪過程圖
    object_map = load_object_map(args.objects_json)
    denoise_label = encode_labels(DENOISING_CONDITION, object_map).unsqueeze(0).to(device)
    gen_denoise = torch.Generator(device="cpu").manual_seed(args.seed + 2)
    _, trajectory = diffusion.sample(
        model=model,
        y=denoise_label,
        image_size=image_size,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        sampler=args.sampler,
        eta=args.eta,
        return_trajectory=True,
        num_trajectory_steps=max(8, args.num_trajectory_steps),
        generator=gen_denoise,
        show_progress=True,
    )
    traj_tensors = denormalize(torch.cat(trajectory, dim=0).cpu())
    denoise_grid = make_grid(traj_tensors, nrow=len(trajectory))
    save_image(denoise_grid, output_dir / "denoising_process.png")

    print(f"\nSaved all images to '{output_dir}':")
    print(f"  - {output_dir / 'test'} (0.png ~ {len(test_labels) - 1}.png)")
    print(f"  - {output_dir / 'new_test'} (0.png ~ {len(new_test_labels) - 1}.png)")
    print(f"  - {output_dir / 'grid_test.png'}")
    print(f"  - {output_dir / 'grid_new_test.png'}")
    print(f"  - {output_dir / 'denoising_process.png'} ({len(trajectory)} steps for {DENOISING_CONDITION})")

    # 6. 使用 pretrained ResNet-18 evaluator 計算 Top-k 準確率
    if not args.skip_eval:
        if device.type != "cuda":
            print("\n[Warning] evaluator.py requires CUDA; pass --skip_eval on CPU.")
            return
        from evaluator import evaluation_model

        evaluator = evaluation_model()
        test_acc = evaluator.eval(prepare_for_evaluator(test_samples), test_labels)
        new_test_acc = evaluator.eval(prepare_for_evaluator(new_test_samples), new_test_labels)

        print("\n========== Evaluation Results ==========")
        print(f"test.json     Top-k Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")
        print(f"new_test.json Top-k Accuracy: {new_test_acc:.4f} ({new_test_acc * 100:.2f}%)")
        print("========================================")


if __name__ == "__main__":
    main()
