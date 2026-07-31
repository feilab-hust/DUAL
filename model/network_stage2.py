import functools
import logging
from torch.nn import init
from model.unet import UNet
from model.diffusion import GaussianDiffusion
####################
# initialize
####################


def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, std)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    logger = logging.getLogger('train')
    logger.info('Initialization method [{:s}]'.format(init_type))
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError(
            'initialization method [{:s}] not implemented'.format(init_type))


def define_G(opt, custom_s1=None):
    if custom_s1 is not None:
        predenoiser = custom_s1['denoiser']
    else:
        predenoiser_opt = opt['s1_model']['unet']
        predenoiser = UNet(
                        in_channel=predenoiser_opt['in_channel'],
                        out_channel=predenoiser_opt['out_channel'],
                        norm_groups=predenoiser_opt['norm_groups'],
                        inner_channel=predenoiser_opt['inner_channel'],
                        channel_mults=predenoiser_opt['channel_multiplier'],
                        attn_res=predenoiser_opt['attn_res'],
                        res_blocks=predenoiser_opt['res_blocks'],
                        dropout=predenoiser_opt.get('dropout', 0.0),
                        image_size=predenoiser_opt['image_size'],
                        version=predenoiser_opt.get('version', 'v2'),
                        with_noise_level_emb=False,
                        padding_size=predenoiser_opt.get('padding_size', 0),
                        padding_mode=predenoiser_opt.get('padding_mode', 'reflect'),
                    )

    model_opt = opt['model']
    init_type = model_opt['unet']['init_type'] if 'init_type' in model_opt['unet'] else 'kaiming'

    diff_denoisor = UNet(
        in_channel=model_opt['unet']['in_channel'],
        out_channel=model_opt['unet']['out_channel'],
        norm_groups=model_opt['unet']['norm_groups'],
        inner_channel=model_opt['unet']['inner_channel'],
        channel_mults=model_opt['unet']['channel_multiplier'],
        attn_res=model_opt['unet']['attn_res'],
        res_blocks=model_opt['unet']['res_blocks'],
        dropout=model_opt['unet']['dropout'],
        image_size=model_opt['unet']['image_size'],
        version=model_opt['unet']['version'],
        padding_size=model_opt['unet']['padding_size'] if 'padding_size' in model_opt['unet'] else 0,
        padding_mode=model_opt['unet']['padding_mode'] if 'padding_mode' in model_opt['unet'] else 'reflect',
    )

    net_s2 = GaussianDiffusion(
                denoisor=diff_denoisor,
                schedule_opt=model_opt['beta_schedule'],
                predenoiser=predenoiser,
                eta=model_opt['eta']
            )
    
    if opt['phase'] == 'train':
        init_weights(net_s2, init_type=init_type)

    return net_s2
