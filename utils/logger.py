import os
import logging
from collections import OrderedDict
import json
from datetime import datetime
import torch
import socket
import numpy as np
import random
import torch.distributed as dist

def mkdirs(paths):
    if isinstance(paths, str):
        os.makedirs(paths, exist_ok=True)
    else:
        for path in paths:
            os.makedirs(path, exist_ok=True)


def get_timestamp():
    return datetime.now().strftime('%y%m%d_%H%M%S')


def parse(args, stage=2):
    assert stage in [1, 2]

    phase = args.phase
    opt_path = args.config
    gpu_ids = args.gpu_ids
    # remove comments starting with '//'
    json_str = ''
    with open(opt_path, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
    opt = json.loads(json_str, object_pairs_hook=OrderedDict)
    opt.update(vars(args))
    opt['exist_gt'] = opt['datasets']['val']['gt_dataroot'] is not None

    # set log directory
    if args.debug:
        opt['name'] = 'debug_{}'.format(opt['name'])
    
    if stage == 1: # train noise model
        # experiments_root = os.path.join(
        #     'experiments', '{}_noisemodel_{}'.format(opt['name'], get_timestamp()))
        experiments_root = os.path.join(
            'experiments', '{}_s1'.format(opt['name']))
    else : # train diffusion model
        assert opt["s1_model"]["resume_state"] != None
        experiments_root = os.path.join(
            'experiments', '{}_s2'.format(opt['name']))
        mkdirs(experiments_root)

        # check match file
        if opt['match_file'] is None:
            match_file_path = os.path.join(experiments_root, f"adaptive_matching.txt")
            if not os.path.isfile(match_file_path):
                with open(match_file_path, 'w') as file:
                    pass
                print(f"[!] adaptive_matching file not exist, created in {match_file_path} ")
            else:
                print(f"[!] adaptive_matching file found in {match_file_path} ")
            opt['match_file'] = match_file_path
        else:
            print(f"[!] adaptive_matching file found in {opt['match_file']} ")


    opt['path']['experiments_root'] = experiments_root

    # create all paths
    for key, path in opt['path'].items():
        if 'resume' not in key and 'experiments' not in key and 'N2N' not in key:
            opt['path'][key] = os.path.join(experiments_root, path)
            mkdirs(opt['path'][key])


    # change dataset length limit
    # opt['phase'] = phase

    if gpu_ids is not None:
        opt['gpu_ids'] = [int(id) for id in gpu_ids.split(',')]
        gpu_list = opt['gpu_ids']
    else:
        gpu_list = ','.join(str(x) for x in opt['gpu_ids'])

    if gpu_ids is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids

    if len(gpu_list) > 1:
        opt['distributed'] = True
        opt['local_rank'] = dist.get_rank()
        opt['world_size'] = dist.get_world_size()
        opt['train']['optimizer']['lr'] *= dist.get_world_size()
    else:
        opt['distributed'] = False
        opt['local_rank'] = 0

    # debug
    if 'debug' in opt['name']:
        opt['datasets']['train']['batch_size'] = 2
        opt['datasets']['val']['batch_size'] = 2

        # opt['datasets']['train']['dataroot'] = opt['datasets']['val']['dataroot']
        opt['train']['val_freq'] = 2
        opt['train']['print_freq'] = 2
        opt['train']['save_checkpoint_freq'] = 3

        # opt['model']['beta_schedule']['train']['n_timestep'] = 10
        # opt['model']['beta_schedule']['val']['n_timestep'] = 10
        # opt['datasets']['train']['data_len'] = 6
        # opt['datasets']['val']['data_len'] = 3

    # validation in train phase
    # if phase == 'train':
    #     opt['datasets']['val']['data_len'] = 3


    return opt


class NoneDict(dict):
    def __missing__(self, key):
        return None


# convert to NoneDict, which return None for missing key.
def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt


def dict2str(opt, indent_l=1):
    '''dict to string for logger'''
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False):
    '''set up logger'''
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s', datefmt='%y-%m-%d %H:%M:%S')
    os.makedirs(root, exist_ok=True)
    log_file = os.path.join(root, '{}.log'.format(phase))
    # fh = logging.FileHandler(log_file, mode='w')
    fh = logging.FileHandler(log_file, mode='a')
    fh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    if screen :
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        l.addHandler(sh)

def val_set(opt_match):
    opt_match["datasets"]['val']['val_volume_idx'] = "all"
    opt_match["datasets"]['val']['val_frame_idx'] = "all"
    opt_match["datasets"]['val']['val_slice_idx'] = "all"
    return opt_match

def ddp_setup(gpu_ids):
    gpu_list = list(map(int, gpu_ids.split(',')))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids

    print('export CUDA_VISIBLE_DEVICES=' + gpu_ids)

    if len(gpu_list) > 1:
        # pytorch>=1.9, Prevent NCCL deadlocks
        os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
        os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'

        local_rank = int(os.environ["LOCAL_RANK"])
        # os.environ["MASTER_ADDR"] = "localhost"
        # if "MASTER_PORT" not in os.environ:
        #     os.environ["MASTER_PORT"] = str(find_free_port())

        torch.distributed.init_process_group(backend="nccl", init_method='env://')
        torch.cuda.set_device(local_rank)

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('localhost', 0))
        _, port = s.getsockname()
        return port

def seed_everything(seed=10):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use muti-gpu
    os.environ['PYTHONHASHSEED'] = str(seed) # set hash seed
