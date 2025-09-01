"""
Quick sanity check: synthetic video dataset + VideoSimCLR model forward.
Run: python sanity_check_video.py
"""

import torch
from torch.utils.data import DataLoader

from video_dataset import SyntheticBouncingBalls
from video_model import VideoSimCLRModel


def main():
    dataset = SyntheticBouncingBalls(length=32, num_frames=8, box_size=96, img_size=224)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

    model = VideoSimCLRModel(base_model="vit_base", projection_dim=128, img_size=224,
                             pretrained=False, temporal_agg="mean")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print("Model ready on", device)

    (xi, xj, y) = next(iter(loader))
    print("Clip shapes:", xi.shape, xj.shape)  # (B, T, C, H, W)

    xi = xi.to(device)
    xj = xj.to(device)
    with torch.no_grad():
        zi = model(xi)
        zj = model(xj)
    print("Projection shapes:", zi.shape, zj.shape)  # (B, D)
    # Simple NT-Xent expects concat along batch dim
    z = torch.cat([zi, zj], dim=0)
    print("Concat z shape:", z.shape)


if __name__ == "__main__":
    main()

