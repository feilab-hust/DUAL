import logging
from collections import OrderedDict
import torch
import torch.nn as nn
import os
from . import unet, network_stage1
from .base_model import BaseModel
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP


logger = logging.getLogger('base')

torch.set_printoptions(precision=10)


class DUALStage1(BaseModel):
    def __init__(self, opt, custom_s1=None):
        super(DUALStage1, self).__init__(opt)
        self.opt = opt
        self.local_rank = opt['local_rank']

        model_opt = opt['s1_model']
        denoisor_opt = model_opt['unet']

        
        if custom_s1 is not None:
            required_keys = ['denoiser', 'opt_s1', 'scheduler']
            missing_keys = [key for key in required_keys if key not in custom_s1]
            assert not missing_keys, f"custom_s1 missing required keys: {missing_keys}"

            self.denoisor = custom_s1['denoiser']
        else:
            self.denoisor = unet.UNet(
                                in_channel=denoisor_opt['in_channel'],
                                out_channel=denoisor_opt['out_channel'],
                                norm_groups=denoisor_opt['norm_groups'],
                                inner_channel=denoisor_opt['inner_channel'],
                                channel_mults=denoisor_opt['channel_multiplier'],
                                attn_res=denoisor_opt['attn_res'],
                                res_blocks=denoisor_opt['res_blocks'],
                                dropout=denoisor_opt.get('dropout', 0.0),
                                image_size=denoisor_opt['image_size'],
                                version=denoisor_opt.get('version', 'v2'),
                                with_noise_level_emb=False,
                                padding_size=denoisor_opt.get('padding_size', 0),
                                padding_mode=denoisor_opt.get('padding_mode', 'reflect'),
                            )

        net_s1 = network_stage1.N2N(
            self.denoisor
        )

        if opt['distributed']:
            self.net_s1 = DDP(net_s1.cuda(), device_ids=[self.local_rank], output_device=self.local_rank)
        else:
            self.net_s1 = self.set_device(net_s1)


        if custom_s1 is not None:
            self.opt_s1 = custom_s1['opt_s1']
            self.scheduler = custom_s1['scheduler']
        else:
            self.opt_s1 = torch.optim.Adam(
                self.net_s1.parameters(), lr=opt["s1_model"]["optimizer"]["lr"])
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt_s1, opt["s1_model"]['n_iter'],
                                                                        eta_min=opt["s1_model"]["optimizer"]["lr"] * 0.01)

        self.log_dict = OrderedDict()
        tbdir = os.path.join(opt['path']['tb_logger'], 'tblog')
        self.tb = SummaryWriter(tbdir)
        self.load_network()
        self.counter = 0

    def feed_data(self, data):
        self.data = self.set_device(data)

    def optimize_parameters(self):
        
        self.opt_s1.zero_grad()

        outputs = self.net_s1(self.data)
        
        l_pix = outputs['total_loss']
        l_pix.backward()


        self.opt_s1.step()
        self.scheduler.step()


        # set log
        self.log_dict['l_pix'] = l_pix.item()

        
    def test(self, continous=False):
        self.net_s1.eval()
        # self.net_s1.train()
        with torch.no_grad(): # TTT
            if isinstance(self.net_s1, nn.DataParallel) or isinstance(self.net_s1, DDP):
                self.denoised = self.net_s1.module.denoise(
                    self.data)
            else:
                self.denoised = self.net_s1.denoise(
                    self.data)
        self.net_s1.train()


    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_LR=True, sample=False):
        out_dict = OrderedDict()
        if sample:
            out_dict['SAM'] = self.SR.detach().float().cpu()
        else:
            out_dict['denoised'] = self.denoised.detach().float().cpu()
            out_dict['Y'] = self.data['Y'].detach().float().cpu()

        return out_dict

    def print_network(self):
        pass

    def save_network(self, epoch, iter_step, save_last_only=False):

        if not save_last_only:
            gen_path = os.path.join(
                self.opt['path']['checkpoint'], 'I{}_E{:04d}_gen.pth'.format(iter_step, epoch))
            opt_path = os.path.join(
                self.opt['path']['checkpoint'], 'I{}_E{:04d}_opt.pth'.format(iter_step, epoch))
        else:
            gen_path = os.path.join(
                self.opt['path']['checkpoint'], 'latest_gen.pth'.format(iter_step, epoch))
            opt_path = os.path.join(
                self.opt['path']['checkpoint'], 'latest_opt.pth'.format(iter_step, epoch))
        # gen
        network = self.net_s1
        if isinstance(self.net_s1, nn.DataParallel) or isinstance(self.net_s1, DDP):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step,
                     'scheduler': None, 'optimizer': None}
        opt_state['optimizer'] = self.opt_s1.state_dict()
        torch.save(opt_state, opt_path)

        logger.info(
            'Saved model in [{:s}] ...'.format(gen_path))

    def load_network(self):
        load_path = self.opt['s1_model']['resume_state']
        if load_path is not None:
            logger.info(
                'Loading pretrained model for G [{:s}] ...'.format(load_path))
            gen_path = '{}_gen.pth'.format(load_path)
            opt_path = '{}_opt.pth'.format(load_path)
            # gen
            network = self.net_s1
            if isinstance(self.net_s1, nn.DataParallel) or isinstance(self.net_s1, DDP):
                network = network.module

            network.load_state_dict(torch.load(
                gen_path), strict=(not self.opt['model']['finetune_norm']))
            # network.load_state_dict(torch.load(
            #     gen_path), strict=False)
            if self.opt['phase'] == 'train':
                # optimizer
                opt = torch.load(opt_path)
                self.opt_s1.load_state_dict(opt['optimizer'])
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']
            else:
                self.ckpt_name = os.path.basename(load_path)
