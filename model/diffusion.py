import math
# import torch
# from h5py._hl import datatype
from torch import nn
from functools import partial
import numpy as np
from tqdm import tqdm
import copy
from model.utils import *

TTT = False # test time training not enabled


def make_beta_schedule(n_timestep, linear_start=1e-4, linear_end=2e-2, warmup_frac = 0.7):
    # rev_warmup_beta
    betas = linear_start * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[n_timestep - warmup_time:] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


# gaussian diffusion trainer class

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoisor,
        schedule_opt=None,
        predenoiser=None,
        eta=0.5,
    ):
        super().__init__()

        self.denoisor = denoisor
        self.predenoiser = predenoiser
        self.eta = eta

        # for TTT
        if TTT:
            optim_params = []
            for k, v in self.named_parameters():
                if k.find('matched_state') >= 0:
                    continue
                if k.find('noise_model_variance') >= 0:
                    continue
                optim_params.append(v)
            print('ttt optimizing params:', len(optim_params))
            self.ttt_opt = torch.optim.Adam(optim_params, lr=1e-4)

        if schedule_opt is not None:
            self.init_noise_schedule(schedule_opt, device=torch.device('cuda:0'))


    def init_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = make_beta_schedule(
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end']
        )
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))
        
    def set_loss(self, device):
        self.mseloss = nn.MSELoss().to(device)
        self.l1loss = nn.L1Loss().to(device)

    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t] * noise


    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * \
            x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]

        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, mask=None, condition_x=None, 
                        mask_condition=None, ttt_opt=None):
        
        b, c, w, h = x.shape
 
        single_noise_level = torch.FloatTensor(
        [self.sqrt_alphas_cumprod_prev[t+1]]).repeat(b, 1).to(x.device)

        if ttt_opt is None:
            with torch.no_grad():
                x_recon = flip_denoise(x, self.denoisor, single_noise_level.repeat(4, 1),
                                       flips=[(False, False), (True, False), (False, True), (True, True)], condition_x=condition_x)
                
        else:
            ttt_opt.zero_grad()

            x_recon = flip_denoise(x, self.denoisor, single_noise_level.expand(4, -1), 
                                   flips = [(False, False), (True, False), (False, True), (True, True)])

            ttt_loss = self.mseloss(x_recon, condition_x.detach())
            ttt_loss.requires_grad = True
            ttt_loss.backward()

            ttt_opt.step()

            self.eval()
            x_recon = flip_denoise(x, self.denoisor, single_noise_level.expand(4, -1), 
                                   flips = [(False, False), (True, False), (False, True), (True, True)])
            self.train()
        

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance, x_recon

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None, mask_condition=None, ttt_opt=None):
        model_mean, model_log_variance, x_recon = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x, mask_condition=mask_condition, ttt_opt=ttt_opt)
        
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return model_mean + noise * (0.5 * model_log_variance).exp(), noise, model_mean, x_recon


    @torch.no_grad()
    def p_sample_loop(self, x_in, continous=False, ttt_opt=None, matched_state=1000):
        x = x_in['Y']
        self.input = x_in['Y']
        x_cond = []
        if x_in.get('time_cond1') is not None:
            x_cond.extend((x_in['time_cond1'], x_in['time_cond2']))
        if x_in.get('space_cond1') is not None:
            x_cond.extend((x_in['space_cond1'], x_in['space_cond2']))
        if x_in.get('global_cond') is not None:
            x_cond.append(x_in['global_cond'])

        img = x
        ret_img = x
        
        ttt = None
        if TTT:
            denoisor_fn_state = copy.deepcopy(self.predenoiser.state_dict())

        x_recon = x_in['Y']

        for i in reversed(range(0, matched_state)):
            img, noise, img_wo_noise, x_recon = self.p_sample(img, i, condition_x = x_cond, ttt_opt=ttt)
            ttt = ttt_opt

            if i == 0:
                ret_img = img

        if TTT:
            self.predenoiser.load_state_dict(denoisor_fn_state)
        
        if continous:
            return ret_img
        else:
            return ret_img

    @torch.no_grad()
    def sample(self, x_in, continous=False):
        matched_state = self.num_timesteps
        x_in['X'] = torch.randn_like(x_in['X'])
        return self.p_sample_loop(x_in, continous, matched_state=matched_state)

    @torch.no_grad()
    def denoise(self, x_in, continous=False, ttt_opt=None):
        matched_state = int(x_in['matched_state'][0].item())
        return self.p_sample_loop(x_in, continous, ttt_opt=ttt_opt, matched_state=matched_state)

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            continuous_sqrt_alpha_cumprod * x_start +
            (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise
        )

    def determine_input_stage(self, x, x_start):
        min_lh = 999
        min_t = -1
        for t in range(sqrt_alphas_cumprod_prev.shape[0]-500):
            noise = data['X'] - sqrt_alphas_cumprod_prev[t] * denoised
            noise_mean = torch.mean(noise)
            noise = noise - noise_mean

            mu, std = norm.fit(noise.cpu().numpy())

            diff = np.abs((1 - sqrt_alphas_cumprod_prev[t]**2).sqrt().cpu().numpy() - std)

            if diff < min_lh:
                min_lh = diff
                min_t = t

    @torch.no_grad()
    def interpolate(self, x, t = None, lams=[0.5]):
        assert x['X'].shape[0] == 2
        x1 = dict(X=x['X'][[0]], Y=x['X'][[0]], condition=x['condition'][[0]], matched_state=x['matched_state'][[0]])
        x1 = self.denoise(x1).unsqueeze(0)

        x2 = dict(X=x['X'][[1]], Y=x['X'][[1]], condition=x['condition'][[1]], matched_state=x['matched_state'][[1]])
        x2 = self.denoise(x2).unsqueeze(0)

        b, *_, device = *x1.shape, x1.device
        t = self.num_timesteps

        assert x1.shape == x2.shape
        t_batched = torch.stack([torch.tensor(self.sqrt_alphas_cumprod_prev[t], device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t_batched), (x1, x2))

        imgs = []
        for lam in lams:
            img = (1 - lam) * xt1 + lam * xt2
            img = img.float()
            for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
                img, noise, img_wo_noise, x_recon = self.p_sample(img, i, condition_x=img, ttt_opt=None)
            imgs.append(img)
        return x['X'][[0]], x['X'][[1]], x1,x2, imgs

    def add_condition(self, input, x_in):
        if x_in.get('time_cond1') is not None:
            input.extend((x_in['time_cond1'], x_in['time_cond2']))
        if x_in.get('space_cond1') is not None:
            input.extend((x_in['space_cond1'], x_in['space_cond2']))
        if x_in.get('global_cond') is not None:
            input.append(x_in['global_cond'])
        return input


    def p_losses(self, x_in, debug=False):
        debug_results = dict()

        image1 = x_in['X']

        pre_denised = x_in.get('pre_denoised', None)
        if pre_denised is None:
            image2 = (x_in['Y'] + 1) / 2.0
            x_start = self.predenoiser(image2).detach()
        else:
            pre_denised = (pre_denised + 1) / 2.0
            x_start = self.predenoiser(pre_denised).detach()
            print('[!] pre-denoised file exists, which used for structure estimation in the stage2')

        x_start = x_start * 2.0 - 1

        [b, c, w, h] = x_start.shape

        matched_state = x_in['matched_state']

        fixed_alphas = []
        for i in range(matched_state.shape[0]):
            fixed_alphas.append(torch.zeros(1,1,1,1).to(x_start.device) + self.sqrt_alphas_cumprod_prev[int(matched_state[i].item())])

        fixed_alphas = torch.cat(fixed_alphas, dim=0)

        t = np.random.randint(1, self.num_timesteps + 1)

        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t-1],
                self.sqrt_alphas_cumprod_prev[t],
                size=b
            )
        ).to(x_start.device)

        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(
            b, -1)
        noise = (image1 - fixed_alphas * x_start.detach()) / (1 - fixed_alphas ** 2).sqrt()

        noise_mean = torch.mean(noise, dim=(1,2,3), keepdim=True)
        noise = noise - noise_mean.detach()
        x_start = x_start + noise_mean.detach() * (1 - fixed_alphas**2).sqrt() / fixed_alphas

        if debug:
            debug_results['noise'] = noise
            debug_results['recon'] = x_start

        noise = noise.view(b, c, -1)
        rand_idx = torch.randperm(noise.shape[-1])
        noise = noise[:,:,rand_idx].view(b,c,w,h).detach()

        x_noisy = self.q_sample(
            x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise.detach())

        input = [x_noisy]
        if x_in.get('time_cond1') is not None:
            input.extend((x_in['time_cond1'],x_in['time_cond2']))
        if x_in.get('space_cond1') is not None:
            input.extend((x_in['space_cond1'],x_in['space_cond2']))
        if x_in.get('global_cond') is not None:
            input.append(x_in['global_cond'])
        x_recon = self.denoisor(torch.cat(input, dim = 1), continuous_sqrt_alpha_cumprod)

        total_loss = self.eta * self.mseloss(x_recon, x_start) + (1 - self.eta) * self.mseloss(x_recon, image1)

        if debug:
            return_dict = dict(total_loss=total_loss, debug_results=debug_results)
        else:
            return_dict = dict(total_loss=total_loss, x_recon=x_recon, x_start=x_start)
        return return_dict

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)
