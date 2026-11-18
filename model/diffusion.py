from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler


class ConditionalDDPM:
    """Conditional DDPM 擴散過程封裝：包含 Noise Schedule、CFG 訓練損失與 DDPM/DDIM 取樣。"""

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_schedule: str = "squaredcos_cap_v2",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        p_uncond: float = 0.1,
        prediction_type: str = "epsilon",
        clip_sample: bool = True,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.p_uncond = p_uncond
        self.prediction_type = prediction_type

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            clip_sample=clip_sample,
            prediction_type=prediction_type,
        )

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion process: q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)."""
        if noise is None:
            noise = torch.randn_like(x_0)
        x_t = self.noise_scheduler.add_noise(x_0, noise, t)
        return x_t, noise

    def drop_condition(self, y: torch.Tensor) -> torch.Tensor:
        """CFG 訓練用：以 p_uncond 機率將單筆樣本的 multi-hot 條件向量清為全 0（unconditional）。"""
        if self.p_uncond <= 0.0:
            return y
        keep_mask = (torch.rand(y.size(0), 1, device=y.device) >= self.p_uncond).to(y.dtype)
        return y * keep_mask

    def training_loss(
        self,
        model: nn.Module,
        x_0: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """計算單步訓練的 MSE 損失（含隨機時間步取樣與 CFG condition dropout）。"""
        batch_size = x_0.size(0)
        device = x_0.device
        y = y.to(device)

        # 1. 為每張圖片隨機抽樣時間步 t ~ Uniform({0, ..., T-1})
        t = torch.randint(
            0,
            self.num_train_timesteps,
            (batch_size,),
            device=device,
            dtype=torch.long,
        )

        # 2. Forward process 加噪得到 x_t
        x_t, noise = self.q_sample(x_0, t)

        # 3. CFG condition dropout
        y_cond = self.drop_condition(y)

        # 4. 模型預測與目標對齊
        pred = model(x_t, t, y_cond)

        if self.prediction_type == "epsilon":
            target = noise
        elif self.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(x_0, noise, t)
        elif self.prediction_type == "sample":
            target = x_0
        else:
            raise ValueError(f"Unsupported prediction_type: {self.prediction_type}")

        return F.mse_loss(pred, target)

    def predict_cfg(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        guidance_scale: float = 3.5,
    ) -> torch.Tensor:
        """使用 Classifier-Free Guidance 計算單步去噪預測值。"""
        if guidance_scale == 1.0:
            return model(x_t, t, y)
        if guidance_scale == 0.0:
            return model(x_t, t, torch.zeros_like(y))

        # 將 unconditional (全 0) 與 conditional 拼成 2B 的 batch 一次前向傳播，減少呼叫開銷
        x_in = torch.cat([x_t, x_t], dim=0)
        t_in = torch.cat([t, t], dim=0)
        y_in = torch.cat([torch.zeros_like(y), y], dim=0)

        pred_uncond, pred_cond = model(x_in, t_in, y_in).chunk(2, dim=0)
        return pred_uncond + guidance_scale * (pred_cond - pred_uncond)

    def _build_sampler(
        self,
        sampler: str,
        num_inference_steps: int,
        device: torch.device,
    ) -> Union[DDPMScheduler, DDIMScheduler]:
        sampler = sampler.lower()
        if sampler == "ddpm":
            scheduler = DDPMScheduler.from_config(self.noise_scheduler.config)
        elif sampler == "ddim":
            scheduler = DDIMScheduler.from_config(self.noise_scheduler.config)
        else:
            raise ValueError(f"Unsupported sampler '{sampler}'. Choose 'ddpm' or 'ddim'.")

        scheduler.set_timesteps(num_inference_steps, device=device)
        return scheduler

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        y: torch.Tensor,
        image_size: int = 64,
        in_channels: int = 3,
        num_inference_steps: Optional[int] = None,
        guidance_scale: float = 3.5,
        sampler: str = "ddim",
        eta: float = 0.0,
        return_trajectory: bool = False,
        num_trajectory_steps: int = 8,
        generator: Optional[torch.Generator] = None,
        show_progress: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, list[torch.Tensor]]]:
        """從純高斯雜訊 x_T 反向去噪生成 [-1, 1] 範圍的圖片。

        Args:
            model: 去噪模型（如 ConditionalUNet）。
            y: (B, num_classes) 的 multi-hot 條件向量。
            image_size: 生成圖片解析度，i-CLEVR 為 64。
            in_channels: 影像通道數，RGB 為 3。
            num_inference_steps: 反向去噪步數（DDIM 預設 100，DDPM 預設 num_train_timesteps）。
            guidance_scale: CFG 引導強度 w；1.0 表示純 conditional，>1.0 增強條件對齊。
            sampler: 取樣器類型，"ddim" 或 "ddpm"。
            eta: 僅 DDIM 有效；0.0 為確定性取樣，1.0 恢復隨機性。
            return_trajectory: 若為 True，額外回傳從雜訊到清晰的去噪過程影格清單。
            num_trajectory_steps: 欲記錄的去噪過程影格數（Lab 要求至少 8 張）。
            generator: 可選的 torch.Generator 以固定隨機種子。
            show_progress: 是否顯示 tqdm 進度條。

        Returns:
            若 return_trajectory=False，回傳 (B, C, H, W) 範圍 [-1, 1] 的生成張量；
            若 return_trajectory=True，回傳 (samples, trajectory)，其中 trajectory 為長度
            num_trajectory_steps 的 list，每個元素為 (B, C, H, W) 範圍 [-1, 1] 的張量。
        """
        sampler_name = sampler.lower()
        if num_inference_steps is None:
            num_inference_steps = 100 if sampler_name == "ddim" else self.num_train_timesteps

        device = next(model.parameters()).device
        y = y.to(device)
        batch_size = y.size(0)

        scheduler = self._build_sampler(sampler_name, num_inference_steps, device)
        timesteps = scheduler.timesteps
        total_steps = len(timesteps)

        # 從標準常態分佈取樣初始雜訊 x_T ~ N(0, I)
        shape = (batch_size, in_channels, image_size, image_size)
        if generator is not None and generator.device != device:
            x = torch.randn(shape, generator=generator, device=generator.device).to(device)
        else:
            x = torch.randn(shape, generator=generator, device=device)

        # 計算要記錄的去噪過程影格索引（包含起點 0 與終點 total_steps）
        trajectory: list[torch.Tensor] = []
        record_indices: set[int] = set()
        if return_trajectory:
            k = min(num_trajectory_steps, total_steps + 1)
            if k <= 1:
                record_indices = {total_steps}
            else:
                record_indices = {round(i * total_steps / (k - 1)) for i in range(k)}
            if 0 in record_indices:
                trajectory.append(x.detach().clamp(-1.0, 1.0))

        was_training = model.training
        model.eval()
        try:
            iterator = enumerate(timesteps, start=1)
            if show_progress:
                from tqdm.auto import tqdm

                iterator = tqdm(iterator, total=total_steps, desc=f"Sampling ({sampler_name.upper()})")

            for step_idx, t in iterator:
                t_batch = torch.full((batch_size,), t.item(), device=device, dtype=torch.long)
                pred = self.predict_cfg(model, x, t_batch, y, guidance_scale=guidance_scale)

                step_kwargs = {"generator": generator}
                if sampler_name == "ddim":
                    step_kwargs["eta"] = eta

                x = scheduler.step(pred, t, x, **step_kwargs).prev_sample

                if return_trajectory and step_idx in record_indices:
                    trajectory.append(x.detach().clamp(-1.0, 1.0))
        finally:
            if was_training:
                model.train()

        samples = x.clamp(-1.0, 1.0)
        if return_trajectory:
            return samples, trajectory
        return samples
