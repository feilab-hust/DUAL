import csv
import os
import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tifffile as tiff
import torch
from pytorch_msssim import ms_ssim
from skimage.metrics import peak_signal_noise_ratio as PSNR


def tensor2img(tensor, out_type=np.uint16, min_max=(-1, 1)):
    """Convert a BCHW/BTCHW model tensor to a NumPy image stack."""
    tensor = tensor.squeeze(1).float().cpu().clamp_(*min_max)
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])
    img_np = tensor.numpy()

    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
    elif out_type == np.uint16:
        img_np = (img_np * 65535.0).round()
    return img_np.astype(out_type)


def numpy2tensor_norm(image):
    image_tensor = torch.from_numpy(image)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return image_tensor.to(dtype=torch.float32, device=device) / 65535.0


def save_img(img, img_path, mode='RGB', slice_norm=False):
    del mode, slice_norm
    tiff.imwrite(img_path, np.squeeze(img), imagej=True)


def calculate_psnr(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(65535.0 / math.sqrt(mse))


def slice_norm(images, datatype='uint16'):
    images = np.asarray(images)
    min_val = np.min(images, axis=(-2, -1), keepdims=True)
    max_val = np.max(images, axis=(-2, -1), keepdims=True)
    images_norm = (images - min_val) / (max_val - min_val + 1e-8)
    if datatype == 'uint16':
        return (images_norm * 65535.0).astype(np.uint16)
    if datatype == 'float32':
        return images_norm.astype(np.float32)
    raise ValueError(f"Unsupported datatype: {datatype}")


class MetricEvaluator:
    """Optional full-reference metric helper used only when GT is configured."""

    def __init__(self):
        self.cache = {
            'denoised_psnr': [],
            'denoised_ssim': [],
            'input_psnr': [],
            'input_ssim': [],
        }

    @staticmethod
    def cacl_psnr(gt, pred):
        gt = slice_norm(gt)
        pred = slice_norm(pred)
        return np.array([
            PSNR(g, p, data_range=65535.0) for g, p in zip(gt, pred)
        ])

    @staticmethod
    def cacl_ms_ssim(gt, pred):
        gt_t = torch.from_numpy(slice_norm(gt, datatype='float32')).unsqueeze(1)
        pred_t = torch.from_numpy(slice_norm(pred, datatype='float32')).unsqueeze(1)
        return ms_ssim(gt_t, pred_t, data_range=1.0, size_average=False).numpy()

    def evaluate_batch(self, gt_batch, denoised_batch, input_batch):
        with ThreadPoolExecutor(max_workers=4) as executor:
            denoised_psnr = executor.submit(self.cacl_psnr, gt_batch, denoised_batch)
            denoised_ssim = executor.submit(self.cacl_ms_ssim, gt_batch, denoised_batch)
            input_psnr = executor.submit(self.cacl_psnr, gt_batch, input_batch)
            input_ssim = executor.submit(self.cacl_ms_ssim, gt_batch, input_batch)

        return {
            'denoised_psnr': denoised_psnr.result(),
            'denoised_ssim': denoised_ssim.result(),
            'input_psnr': input_psnr.result(),
            'input_ssim': input_ssim.result(),
        }

    def aggregate_results(self):
        def average(key):
            values = self.cache[key]
            return float(np.mean(values)) if values else float('nan')

        return {
            'denoised': {
                'psnr_avr': f"{average('denoised_psnr'):.4f}",
                'ssim_avr': f"{average('denoised_ssim'):.4f}",
            },
            'input': {
                'psnr_avr': f"{average('input_psnr'):.4f}",
                'ssim_avr': f"{average('input_ssim'):.4f}",
            },
        }

    def _write_metric_to_csv(self, csv_file, headers, data_key1, data_key2, img_shape):
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            count = 0
            for i in range(img_shape[0]):
                for j in range(img_shape[1]):
                    for k in range(img_shape[2]):
                        writer.writerow([
                            f'n_{i}_t_{j}_z_{k}',
                            self.cache[data_key1][count],
                            self.cache[data_key2][count],
                        ])
                        count += 1

    def save_csv(self, csv_path, img_shape, header=None):
        os.makedirs(csv_path, exist_ok=True)
        denoised_psnr_header = header if header is not None else 'denoised_psnr'
        denoised_ssim_header = header if header is not None else 'denoised_ssim'
        self._write_metric_to_csv(
            os.path.join(csv_path, 'psnr.csv'),
            ['idx', 'input_psnr', denoised_psnr_header],
            'input_psnr', 'denoised_psnr', img_shape,
        )
        self._write_metric_to_csv(
            os.path.join(csv_path, 'ssim.csv'),
            ['idx', 'input_ssim', denoised_ssim_header],
            'input_ssim', 'denoised_ssim', img_shape,
        )
