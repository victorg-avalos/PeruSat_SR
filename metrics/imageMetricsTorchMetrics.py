import torch
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure

import torch
from torchmetrics.functional.image import (
    learned_perceptual_image_patch_similarity,
    multiscale_structural_similarity_index_measure,
    spectral_angle_mapper
)
from torchmetrics.functional import mean_absolute_error


def compute_quality_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    device: torch.device = None,
    mode: str = 'rgb'  # 'rgb' or 'ms'
) -> torch.Tensor:
    """
    Combined loss for RGB or multispectral SR including:
      - L1 (MAE) loss
      - MS‑SSIM loss
      - LPIPS (if mode='rgb') or SAM (if mode='ms')

    Returns a single scalar loss value.
    """
    if device is not None:
        pred = pred.to(device)
        target = target.to(device)

    # L1 loss
    l1 = mean_absolute_error(pred, target)

    # MS-SSIM loss across all channels
    ms_ssim_val = multiscale_structural_similarity_index_measure(
        pred, target, data_range=data_range
    )
    ms_ssim_loss = 1.0 - ms_ssim_val  # Multi-scale SSIM loss :contentReference[oaicite:1]{index=1}

    # Mode-specific loss
    if mode == 'rgb':
        pred_norm = pred.clamp(0.0, 1.0)
        target_norm = target.clamp(0.0, 1.0)
        #lpips = learned_perceptual_image_patch_similarity(
            #pred_norm, target_norm,
            #net_type='vgg',
            #normalize=True
        #)
        total_loss = l1 + ms_ssim_loss# + lpips
    elif mode == 'ms':
        sam = spectral_angle_mapper(pred, target)
        total_loss = l1 + ms_ssim_loss + sam
    else:
        raise ValueError("mode must be 'rgb' or 'ms'")

    return total_loss

def compute_quality_losses(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    use_psnr: bool = False,
    use_ssim: bool = False,
    device: torch.device = None
) -> dict:
    """
    Compute PSNR y/o SSIM como funciones de pérdida de manera diferenciable en GPU.

    Args:
        pred        (B, C, H, W)-tensor: salida del modelo (super-resolución).
        target      (B, C, H, W)-tensor: imagen de referencia.
        data_range   float: valor máximo de píxel (p.ej. 1.0 o 255).
        use_psnr     bool: Si True, calcular pérdida basada en PSNR = -PSNR(pred, target).
        use_ssim     bool: Si True, calcular pérdida basada en SSIM (1 - SSIM promedio por canal).
        device    torch.device: dispositivo donde ejecutar (p.ej. torch.device("cuda") o "cpu").

    Returns:
        losses dict con claves:
            'psnr_loss' (si use_psnr=True) → -PSNR (maximizar PSNR al minimizar pérdida)
            'ssim_loss' (si use_ssim=True) → 1 - SSIM_promedio_por_canal
    """
    if device is not None:
        pred = pred.to(device)
        target = target.to(device)

    losses = {}

    if use_psnr:
        # PSNR ahora importado desde torchmetrics.image
        psnr_val = peak_signal_noise_ratio(pred, target, data_range=data_range)
        losses['psnr_loss'] = -psnr_val/40

    if use_ssim:
        B, C, H, W = pred.shape
        ssim_sum = torch.zeros(C, device=device or pred.device)

        for c in range(C):
            pred_c = pred[:, c:c+1, :, :]
            target_c = target[:, c:c+1, :, :]
            ssim_val_c = structural_similarity_index_measure(
                pred_c, target_c, data_range=data_range
            )
            ssim_sum[c] = ssim_val_c

        ssim_mean_channels = ssim_sum.mean()
        losses['ssim_loss'] = 1.0 - ssim_mean_channels

    return losses

def compute_metrics_for_model_dataset(
    generator: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    data_range: float = 1.0,
    device: torch.device = None,
    per_channel: bool = False
) -> dict:
    """
    Aplica un modelo generador a un DataLoader y calcula PSNR/SSIM.
    Si per_channel es True, devuelve métricas por canal y sus promedios.
    Si es False, calcula una métrica agregada sobre todos los canales.

    Args:
        generator   (nn.Module): modelo generador (e.g., super-resolución).
        dataloader  (DataLoader): iterador que devuelve tuplas (lr, hr).
        data_range   (float): valor máximo de píxel (p.ej. 1.0 o 255).
        device      (torch.device): dispositivo para inferencia (p.ej. "cuda").
        per_channel (bool): True para cálculo por canal, False para agregación multicanal.

    Returns:
        dict: si per_channel:
                  'psnr_per_channel': tensor (C,)
                  'ssim_per_channel': tensor (C,)
                  'psnr_overall_mean': escalar
                  'ssim_overall_mean': escalar
              else:
                  'psnr': escalar
                  'ssim': escalar
    """
    # Preparar modelo en modo eval y en el dispositivo
    if device is not None:
        generator = generator.to(device)
    generator.eval()

    total_images = 0
    if per_channel:
        sum_psnr = None
        sum_ssim = None
    else:
        sum_psnr = 0.0
        sum_ssim = 0.0

    with torch.no_grad():
        for lr_batch, hr_batch in dataloader:
            # Mover a GPU/CPU
            if device is not None:
                lr_batch = lr_batch.to(device)
                hr_batch = hr_batch.to(device)

            # Inferencia
            sr_batch = generator(lr_batch)
            B, C, H, W = sr_batch.shape

            if per_channel:
                # Inicializar vectores en el primer batch
                if sum_psnr is None:
                    sum_psnr = torch.zeros(C, device=sr_batch.device)
                    sum_ssim = torch.zeros(C, device=sr_batch.device)
                # Calcular por canal
                for c in range(C):
                    sr_c = sr_batch[:, c:c+1, :, :]
                    hr_c = hr_batch[:, c:c+1, :, :]
                    psnr_val = peak_signal_noise_ratio(sr_c, hr_c, data_range=data_range)
                    #print(f"channel_{c}_PSNR={psnr_val}")
                    ssim_val = structural_similarity_index_measure(sr_c, hr_c, data_range=data_range)
                    #print(f"channel_{c}_SSIM={ssim_val}")
                    sum_psnr[c] += psnr_val * B
                    sum_ssim[c] += ssim_val * B
            else:
                # Cálculo agregado multicanal
                psnr_val = peak_signal_noise_ratio(sr_batch, hr_batch, data_range=data_range)
                #ssim_val = structural_similarity_index_measure(sr_batch, hr_batch, data_range=data_range)
                ssim_val = multiscale_structural_similarity_index_measure(sr_batch, hr_batch, data_range=data_range)
                sum_psnr += psnr_val * B
                sum_ssim += ssim_val * B

            total_images += B

    # Compilar resultados
    if per_channel:
        psnr_per_channel = sum_psnr / total_images
        ssim_per_channel = sum_ssim / total_images
        return {
            'psnr_per_channel': psnr_per_channel,
            'ssim_per_channel': ssim_per_channel,
            'psnr_overall_mean': psnr_per_channel.mean(),
            'ssim_overall_mean': ssim_per_channel.mean()
        }
    else:
        psnr = sum_psnr / total_images
        ssim = sum_ssim / total_images
        return {
            'psnr': psnr,
            'ssim': ssim
        }