import torch
import logging
import numpy as np

from tqdm import tqdm
import torch.distributed as dist
from model.model_stage1 import DUALStage1


def _rev_warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):    # NOTE: 生成βt
    betas = linear_start * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[n_timestep - warmup_time:] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas

def adapt_step(loader, opt, custom_s1=None):
    logger = logging.getLogger(opt['phase'])
    rank = opt['local_rank']
    if rank == 0:
        logger.info('Markov chain state matching!')

    # model
    trainer = DUALStage1(opt, custom_s1=custom_s1)
    trainer.net_s1.eval()

    betas = _rev_warmup_beta(opt['s1_model']['beta_schedule']['linear_start'], opt['s1_model']['beta_schedule']['linear_end'],
                                 1000, 0.7)
    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod_prev = np.sqrt(np.append(1., alphas_cumprod))   # NOTE: sqrt_alphas_cumprod_prev[t]  = √ᾱt
    sqrt_alphas_cumprod_prev = torch.tensor(sqrt_alphas_cumprod_prev).cuda()

    sqrt_one_minus_sq = torch.sqrt(1. - sqrt_alphas_cumprod_prev ** 2)  # NOTE: sqrt_one_minus_sq[t]         = √(1−ᾱt)
    t_params = torch.stack([sqrt_alphas_cumprod_prev, sqrt_one_minus_sq], dim=1)  # (T, 2)

    world_size = dist.get_world_size() if opt['distributed'] else 1
    total_samples = len(loader.dataset)
    samples_per_rank = (total_samples + world_size - 1) // world_size
    start_idx = rank * samples_per_rank

    img_shape = loader.dataset.images_shape
    all_results = []

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(loader, desc='Match state', disable=opt['local_rank']!=0)):

            data = trainer.set_device(data)

            data['Y'] = (data['Y'] + 1) / 2.0
            X = data['X']   # NOTE: 公式中的观测图像 y，训练阶段是两张子图之一 IA，测试阶段X=Y
            if opt['distributed']:
                denoised = trainer.net_s1.module.denoise(data)  # (B,C,H,W)
            else:
                denoised = trainer.net_s1.denoise(data) # NOTE: 第一阶段结构估计^x0，训练阶段由另一张子图IB进过回归模型生成

            denoised = denoised * 2 - 1

            scaled_denoised = denoised.unsqueeze(1) * t_params[:, 0].view(1, -1, 1, 1, 1)  # NOTE: (B,T,C,H,W)    t_params[:, 0]: 所有候选时间步的√ᾱt
            noise = X.unsqueeze(1) - scaled_denoised  # NOTE: (B,T,C,H,W)   所有候选时间步对应的rt = y - √ᾱt×^x0

            noise_centered = noise - noise.mean(dim=(2, 3, 4), keepdim=True)    # NOTE: 残差均值中心化 rtc = rt - E[rt]
            stds = noise_centered.std(dim=(2, 3, 4))  # NOTE: (B,T) 计算残差标准差 st = Std(rtc)

            diffs = torch.abs(sqrt_one_minus_sq.view(1, -1) - stds)  # NOTE: (B,T)  与理论累计噪声强度比较 dt = |Std(rtc) - √1-ᾱt|

            min_t_indices = []  # NOTE: 选择最佳时间步t*   argmin|Std(rtc) - √1-ᾱt|
            for b in range(diffs.size(0)):
                min_t = -1
                min_diff = float('inf')
                prev_diff = float('inf')
                for t in range(diffs.size(1)):
                    current_diff = diffs[b, t].item()
                    if current_diff < min_diff:
                        min_diff = current_diff
                        min_t = t
                    # Early Terminate
                    if current_diff > prev_diff and t > 10: # PS: 代码没有直接调用 argmin()，而是逐步扫描，并在差值开始增大后提前终止。它假定差值随时间步大致先下降后上升。
                        break
                    prev_diff = current_diff
                min_t_indices.append(min_t)


            batch_start = start_idx + batch_idx * loader.batch_size
            batch_indices = [batch_start + i for i in range(len(min_t_indices))]

            batch_results = [
                f"{idx // (img_shape[1] * img_shape[2])}_{(idx % (img_shape[1] * img_shape[2])) // img_shape[2]}_{idx % img_shape[2]}_{t}"
                for idx, t in zip(batch_indices, min_t_indices)
            ]
            all_results.extend(batch_results)

            torch.cuda.empty_cache()

    if opt['distributed']:
        gathered_results = [None] * world_size
        dist.all_gather_object(gathered_results, all_results)

        if dist.get_rank() == 0:
            combined = []
            for res in gathered_results:
                combined.extend(res)

            combined.sort(key=lambda x: int(x.split('_')[0]))
            with open(opt['match_file'], 'w') as f:
                f.write('\n'.join(combined))

        dist.barrier()
    else:
        with open(opt['match_file'], 'w') as f:
            f.write('\n'.join(all_results))     # NOTE: 保存每张图像的t*，保存格式为volume_frame_slice_t。例如：0_25_3_417，表示第0个 volume、第25帧、第3个切片，对应t*=417

