import argparse
import copy
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model.diffusion import ConditionalDDPM
from model.unet import ConditionalUNet
from src.dataset import build_dataloader, load_condition_file


class EMA:
    """Exponential Moving Average (EMA) 權重維護器，含初期 warmup decay。"""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval()
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module, step: int) -> None:
        decay = min(self.decay, (1.0 + step) / (10.0 + step))
        for ema_param, param in zip(self.ema_model.parameters(), model.parameters()):
            ema_param.lerp_(param.detach(), 1.0 - decay)
        for ema_buf, buf in zip(self.ema_model.buffers(), model.buffers()):
            ema_buf.copy_(buf)

    def state_dict(self) -> dict:
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self.ema_model.load_state_dict(state_dict)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.05,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup + Cosine decay 學習率排程。"""

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    path: Path,
    epoch: int,
    global_step: int,
    model: nn.Module,
    ema: EMA,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    best_acc: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": lr_scheduler.state_dict(),
            "best_acc": best_acc,
            "config": {
                "num_classes": 24,
                "sample_size": args.image_size,
                "block_out_channels": [128, 128, 256, 512],
                "num_train_timesteps": args.num_train_timesteps,
                "beta_schedule": args.beta_schedule,
                "p_uncond": args.p_uncond,
                "prediction_type": args.prediction_type,
            },
        },
        path,
    )


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    diffusion: ConditionalDDPM,
    evaluator,
    test_labels: torch.Tensor,
    new_test_labels: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    preview_path: Path,
) -> tuple[float, float, float]:
    """用 EMA 模型生成 test / new_test 圖片並透過 pretrained ResNet-18 評估準確率。"""
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    test_images = diffusion.sample(
        model=model,
        y=test_labels.to(device),
        image_size=args.image_size,
        num_inference_steps=args.eval_steps,
        guidance_scale=args.guidance_scale,
        sampler=args.eval_sampler,
        generator=generator,
    )
    new_test_images = diffusion.sample(
        model=model,
        y=new_test_labels.to(device),
        image_size=args.image_size,
        num_inference_steps=args.eval_steps,
        guidance_scale=args.guidance_scale,
        sampler=args.eval_sampler,
        generator=generator,
    )

    test_acc = evaluator.eval(test_images, test_labels)
    new_test_acc = evaluator.eval(new_test_images, new_test_labels)
    avg_acc = 0.5 * (test_acc + new_test_acc)

    # 儲存驗證預覽圖（8x4 grid）方便追蹤訓練進展
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    grid = make_grid((test_images.clamp(-1.0, 1.0) + 1.0) / 2.0, nrow=8)
    save_image(grid, preview_path)

    return test_acc, new_test_acc, avg_acc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Conditional DDPM with CFG and EMA on i-CLEVR")
    parser.add_argument("--data_dir", type=str, default="./data/iclevr")
    parser.add_argument("--train_json", type=str, default="./train.json")
    parser.add_argument("--test_json", type=str, default="./test.json")
    parser.add_argument("--new_test_json", type=str, default="./new_test.json")
    parser.add_argument("--objects_json", type=str, default="./objects.json")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")

    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--no_flip", action="store_true", help="停用訓練時隨機水平翻轉")

    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--beta_schedule", type=str, default="squaredcos_cap_v2")
    parser.add_argument("--prediction_type", type=str, default="epsilon")
    parser.add_argument("--p_uncond", type=float, default=0.1, help="CFG 條件 dropout 機率")
    parser.add_argument("--ema_decay", type=float, default=0.999)

    parser.add_argument(
        "--guidance_scale",
        "--cfg_scale",
        "--cfg_Scale",
        dest="guidance_scale",
        type=float,
        default=3.5,
        help="驗證取樣時使用的 CFG guidance scale",
    )
    parser.add_argument("--eval_every", type=int, default=10, help="每隔幾個 epoch 執行一次 evaluator 驗證")
    parser.add_argument("--eval_steps", type=int, default=100, help="驗證取樣步數")
    parser.add_argument("--eval_sampler", type=str, default="ddim", choices=["ddim", "ddpm"])
    parser.add_argument("--save_every", type=int, default=25, help="每隔幾個 epoch 額外保留一份歷史權重")

    parser.add_argument("--no_amp", action="store_true", help="停用 CUDA 混合精度訓練")
    parser.add_argument("--resume", type=str, default=None, help="從既有 checkpoint 接續訓練")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and (not args.no_amp)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. DataLoader
    train_loader = build_dataloader(
        data_dir=args.data_dir,
        train_json=args.train_json,
        objects_json=args.objects_json,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
        random_flip=not args.no_flip,
        pin_memory=(device.type == "cuda"),
    )

    # 2. Model, Diffusion, EMA
    model = ConditionalUNet(num_classes=24, sample_size=args.image_size).to(device)
    ema = EMA(model, decay=args.ema_decay)
    diffusion = ConditionalDDPM(
        num_train_timesteps=args.num_train_timesteps,
        beta_schedule=args.beta_schedule,
        p_uncond=args.p_uncond,
        prediction_type=args.prediction_type,
    )

    # 3. Optimizer & LR Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    total_steps = args.epochs * len(train_loader)
    lr_scheduler = build_lr_scheduler(optimizer, args.warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 1
    global_step = 0
    best_acc = -1.0

    # 4. Optional Resume
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        ema.load_state_dict(ckpt["ema_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        lr_scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        best_acc = ckpt.get("best_acc", -1.0)
        print(f"Resumed from {args.resume} (epoch {ckpt['epoch']}, best_acc={best_acc:.4f})")

    # 5. Optional Evaluator (evaluator.py 需要 CUDA 與 ./checkpoint.pth)
    evaluator = None
    test_labels = None
    new_test_labels = None
    if args.eval_every > 0 and device.type == "cuda" and Path("./checkpoint.pth").exists():
        from evaluator import evaluation_model

        evaluator = evaluation_model()
        test_labels = load_condition_file(args.test_json, args.objects_json)
        new_test_labels = load_condition_file(args.new_test_json, args.objects_json)
    elif args.eval_every > 0:
        print("[Warning] Skip periodic evaluation (requires CUDA and ./checkpoint.pth).")

    # 6. Training Loop
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)

        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = diffusion.training_loss(model, images, labels)

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

            global_step += 1
            ema.update(model, global_step)

            running_loss += loss.item()
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch:03d}/{args.epochs:03d}] | Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        # 儲存最新權重
        save_checkpoint(
            output_dir / "latest.pth",
            epoch,
            global_step,
            model,
            ema,
            optimizer,
            lr_scheduler,
            best_acc,
            args,
        )

        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(
                output_dir / f"epoch_{epoch:03d}.pth",
                epoch,
                global_step,
                model,
                ema,
                optimizer,
                lr_scheduler,
                best_acc,
                args,
            )

        # 定期以 EMA 權重做驗證並更新 best_model.pth
        if evaluator is not None and (epoch % args.eval_every == 0 or epoch == args.epochs):
            test_acc, new_test_acc, avg_acc = evaluate_model(
                model=ema.ema_model,
                diffusion=diffusion,
                evaluator=evaluator,
                test_labels=test_labels,
                new_test_labels=new_test_labels,
                args=args,
                device=device,
                preview_path=output_dir / "previews" / f"epoch_{epoch:03d}_test_grid.png",
            )
            print(
                f"  -> Eval | test.json: {test_acc:.4f} | "
                f"new_test.json: {new_test_acc:.4f} | avg: {avg_acc:.4f}"
            )
            if avg_acc > best_acc:
                best_acc = avg_acc
                save_checkpoint(
                    output_dir / "best_model.pth",
                    epoch,
                    global_step,
                    model,
                    ema,
                    optimizer,
                    lr_scheduler,
                    best_acc,
                    args,
                )
                print(f"  -> Saved new best_model.pth (avg_acc={best_acc:.4f})")


if __name__ == "__main__":
    main()
