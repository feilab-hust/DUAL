import logging
from collections import OrderedDict
import torch
import torch.nn as nn
import os
import model.network_stage2 as networks
from .base_model import BaseModel
from tensorboardX import SummaryWriter

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

torch.set_printoptions(precision=10)


class DUALStage2(BaseModel):
    def __init__(self, opt, custom_s1=None):
        super(DUALStage2, self).__init__(opt)
        self.opt = opt
        self.local_rank = opt['local_rank']
        self.logger = logging.getLogger(opt['phase'])
        # TTT
        if 'TTT' in opt:
            self.use_ttt = True
        else:
            self.use_ttt = False

        # define network and load pretrained models
        net_s1 = networks.define_G(opt, custom_s1)

        if opt['distributed']:
            dist.barrier()
            self.net_s1 =  torch.nn.SyncBatchNorm.convert_sync_batchnorm(net_s1)
            self.net_s1 = DDP(net_s1.cuda(), device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True)
        else:
            self.net_s1 = self.set_device(net_s1)

        if isinstance(self.net_s1, nn.DataParallel) or isinstance(self.net_s1, DDP):
            self.net_s1_core = self.net_s1.module
        else:
            self.net_s1_core = self.net_s1


        self.loss_type = opt['model'].get('loss_type', None)

        tbdir = os.path.join(opt['path']['tb_logger'], 'tblog')
        self.tb = SummaryWriter(tbdir)

        # set loss and load resume state
        # self.set_loss()   # fixme: 代码调用了一个不存在的 set_loss()方法。DUALStage2 类没有 set_loss()，真正的方法定义在内部的 GaussianDiffusion 模型中
        self.net_s1_core.set_loss(self.device)

        self.init_noise_schedule(opt['model']['beta_schedule'])

        if self.opt['phase'] == 'train':
            self.net_s1.train()
            # find the parameters to optimize
            if opt['model']['finetune_norm']:
                optim_params = []
                for k, v in self.net_s1.named_parameters():
                    v.requires_grad = False
                    if k.find('transformer') >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        self.logger.info(
                            'Params [{:s}] initialized to 0 and will optimize.'.format(k))
            else:
                optim_params = []
                for k, v in self.net_s1.named_parameters():
                    if k.find('predenoiser') >= 0:
                        continue
                    if k.find('noise_model_variance') >= 0:
                        continue
                    optim_params.append(v)
            print('Optimizing: '+str(len(optim_params))+' params')
            
            self.opt_s1 = torch.optim.Adam(
                optim_params, lr=opt['train']["optimizer"]["lr"])

            self.log_dict = OrderedDict()
        
        self.load_network()
        self.counter = 0

    def feed_data(self, data):
        self.data = self.set_device(data)

    def optimize_parameters(self):
        self.opt_s1.zero_grad()

        outputs = self.net_s1(self.data)
        if torch.is_tensor(outputs):
            l_pix = outputs
            l_pix.backward()
            self.opt_s1.step()

        elif type(outputs) is dict:
            l_pix = outputs['total_loss']

            total_loss = l_pix
            total_loss.backward()
            self.opt_s1.step()
    
        self.log_dict['l_pix'] = l_pix.item()

    def test(self, continous=False):
        if self.use_ttt:
            optim_params = []
            for k, v in self.net_s1.named_parameters():
                if k.find('predenoiser') >= 0:
                    continue
                optim_params.append(v)
            
            ttt_opt = torch.optim.Adam(
                optim_params, lr=self.opt['TTT']["optimizer"]["lr"])
        else:
            self.net_s1.eval()
            ttt_opt = None
        with torch.no_grad():
            self.denoised = self.net_s1_core.denoise(self.data, continous, ttt_opt=ttt_opt)
        self.net_s1.train()

    def sample(self, batch_size=1, continous=False):
        self.net_s1.eval()
        with torch.no_grad():
            self.denoised = self.net_s1_core.denoise(self.data, continous)
        self.net_s1.train()

    def interpolate(self, continous=False, lams=[0.5]):
        self.net_s1.eval()
        with torch.no_grad():
            self.denoised = self.net_s1_core.denoise(self.data, continous, lams=lams)
        self.net_s1.train()

    def init_noise_schedule(self, schedule_opt):
        self.net_s1_core.init_noise_schedule(schedule_opt, self.device)

    def set_noise_schedule(self, schedule_opt):
        self.net_s1_core.set_new_noise_schedule(schedule_opt, self.device)

    def get_current_log(self):
        return self.log_dict


    def get_current_visuals(self, need_LR=True, sample=False, interpolate=False):
        out_dict = OrderedDict()
        rank = self.opt.get("local_rank", 0)

        out = self.denoised.detach()
        out_dict['Y'] = self.data['Y'].detach()

        if rank == 0:
            out_dict['denoised'] = out.float().cpu()
            out_dict['Y'] = out_dict['Y'].float().cpu()
        else:
            out_dict['denoised'] = out
            out_dict['Y'] = out_dict['Y']

        return out_dict

    def print_network(self):
        pass

    def save_network(self, epoch, iter_step, save_last_only=False):

        if not save_last_only:
            gen_path = os.path.join(
                self.opt['path']['checkpoint'], 'I{}_E{}_gen.pth'.format(iter_step, epoch))
            opt_path = os.path.join(
                self.opt['path']['checkpoint'], 'I{}_E{}_opt.pth'.format(iter_step, epoch))
        else:
            gen_path = os.path.join(
                self.opt['path']['checkpoint'], 'latest_gen.pth'.format(iter_step, epoch))
            opt_path = os.path.join(
                self.opt['path']['checkpoint'], 'latest_opt.pth'.format(iter_step, epoch))
        network = self.net_s1_core
        
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        opt_state = {'epoch': epoch, 'iter': iter_step,
                     'scheduler': None, 'optimizer': None}
        opt_state['optimizer'] = self.opt_s1.state_dict()
        torch.save(opt_state, opt_path)

        self.logger.info(
            'Saved model in [{:s}] ...'.format(gen_path))

    def load_network(self):
        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            self.logger.info(
                'Loading stage2 pretrained model for G [{:s}] ...'.format(load_path))
            gen_path = '{}_gen.pth'.format(load_path)
            opt_path = '{}_opt.pth'.format(load_path)

            network = self.net_s1
            if isinstance(self.net_s1, nn.DataParallel) or isinstance(self.net_s1, DDP):
                network = network.module
            network.load_state_dict(torch.load(
                gen_path), strict=False)
            if self.opt['phase'] == 'train':
                opt = torch.load(opt_path)
                self.opt_s1.load_state_dict(opt['optimizer'])
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']
            else:
                self.ckpt_name = os.path.basename(load_path)
        elif self.opt['s1_model']['resume_state'] is not None:
            load_path = self.opt['s1_model']['resume_state']
            gen_path = '{}_gen.pth'.format(load_path)
            state_dict = torch.load(gen_path)
            self.logger.info(
                'Loading stage1 pretrained model for G [{:s}] ...'.format(gen_path))
            if isinstance(self.net_s1, nn.DataParallel) or isinstance(self.net_s1, DDP):
                self.net_s1.module.load_state_dict(state_dict, strict=False)
            else:
                self.net_s1.load_state_dict(state_dict, strict=False)
