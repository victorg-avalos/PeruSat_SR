import torch
import numpy as np

# Differentiable losses
from torchmetrics.functional import mean_absolute_error
from torchmetrics.functional.image import (
    multiscale_structural_similarity_index_measure,
    spectral_angle_mapper
)

# SEWAR (non-differentiable) metrics/losses
from sewar.full_ref import psnr as sewar_psnr, ssim as sewar_ssim, msssim as sewar_msssim


def compute_quality_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    device: torch.device = None,
    mode: str = 'rgb'  # 'rgb' or 'ms'
) -> torch.Tensor:
    """
    Combined (differentiable) loss for RGB or multispectral SR including:
      - L1 (MAE)
      - MS-SSIM
      - SAM (if mode='ms')
    """
    if device is not None:
        pred, target = pred.to(device), target.to(device)

    # L1
    l1 = mean_absolute_error(pred, target)

    # MS-SSIM
    ms_ssim_val = multiscale_structural_similarity_index_measure(
        pred, target, data_range=data_range
    )
    ms_ssim_loss = 1.0 - ms_ssim_val

    if mode == 'rgb':
        return l1 + ms_ssim_loss
    elif mode == 'ms':
        sam = spectral_angle_mapper(pred, target)
        return l1 + ms_ssim_loss + sam
    else:
        raise ValueError("mode must be 'rgb' or 'ms'")


def compute_quality_losses(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    use_psnr: bool = False,
    use_ssim: bool = False,
    device: torch.device = None
) -> dict:
    """
    Compute non-diff PSNR/SSIM “losses” over the full multi-channel image via SEWAR.
    Returns:
      - 'psnr_loss' = -(avg_PSNR) / 40   (if use_psnr)
      - 'ssim_loss' = 1 - (avg_SSIM)     (if use_ssim)
    """
    if device is not None:
        pred, target = pred.to(device), target.to(device)

    losses = {}

    # to CPU numpy, [0,1]→[0,MAX]
    pred_np   = (pred.detach().cpu().numpy()   * data_range)
    target_np = (target.detach().cpu().numpy() * data_range)
    B, C, H, W = pred_np.shape

    if use_psnr:
        total = 0.0
        for i in range(B):
            # shape (H,W,C)
            gt = np.transpose(target_np[i], (1, 2, 0))
            pd = np.transpose(pred_np[i],   (1, 2, 0))
            total += sewar_psnr(gt, pd, MAX=data_range)
        avg = total / B
        losses['psnr_loss'] = -avg / 40.0

    if use_ssim:
        total = 0.0
        for i in range(B):
            gt = np.transpose(target_np[i], (1, 2, 0))
            pd = np.transpose(pred_np[i],   (1, 2, 0))
            total += sewar_ssim(gt, pd, MAX=data_range)
        avg = total / B
        losses['ssim_loss'] = 1.0 - avg

    return losses


def compute_metrics_for_model_dataset(
    generator: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    data_range: float = 1.0,
    device: torch.device = None
) -> dict:
    """
    Run the SR model over the DataLoader and compute:
      - 'psnr': avg SEWAR PSNR over all images & channels
      - 'ssim': avg SEWAR MS-SSIM over all images & channels
    """
    if device is not None:
        generator = generator.to(device)
    generator.eval()

    sum_psnr = 0.0
    sum_ssim = 0.0
    count   = 0

    with torch.no_grad():
        for lr, hr in dataloader:
            if device is not None:
                lr, hr = lr.to(device), hr.to(device)

            sr = generator(lr)

            # to CPU numpy, scale
            sr_np = (sr.cpu().numpy() * data_range)
            hr_np = (hr.cpu().numpy() * data_range)
            B = sr_np.shape[0]

            for i in range(B):
                gt = np.transpose(hr_np[i], (1, 2, 0))  # (H,W,C)
                pd = np.transpose(sr_np[i], (1, 2, 0))
                sum_psnr += sewar_psnr(gt, pd, MAX=data_range)
                sum_ssim += sewar_msssim(gt, pd, MAX=data_range)
                count   += 1

    return {
        'psnr': torch.tensor(sum_psnr / count),
        'ssim': torch.tensor(sum_ssim / count)
    }
