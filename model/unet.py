import torch
import torch.nn as nn
from diffusers import UNet2DModel


class ConditionalUNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 24,
        sample_size: int = 64,
        block_out_channels: tuple = (128, 128, 256, 512),
    ):
        super().__init__()
        # UNet2DModel 的 timestep embedding 維度固定為 block_out_channels[0] * 4
        time_embed_dim = block_out_channels[0] * 4

        self.label_emb = nn.Sequential(
            nn.Linear(num_classes, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.unet = UNet2DModel(
            sample_size=sample_size,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=block_out_channels,
            down_block_types=(
                "DownBlock2D",       # 64 -> 32
                "DownBlock2D",       # 32 -> 16
                "AttnDownBlock2D",   # 16 -> 8
                "AttnDownBlock2D",   # 8 -> 8
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
            class_embed_type="identity",
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # y: (B, 24) multi-hot float tensor；CFG unconditional 時傳全 0 向量即可
        cond = self.label_emb(y.to(dtype=x.dtype))
        return self.unet(x, t, class_labels=cond).sample