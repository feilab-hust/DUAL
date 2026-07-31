'''create dataset and dataloader'''
import logging
from torch.utils.data import DistributedSampler,DataLoader
import torch.distributed as dist
from data.dataset import S1Dataset, S2Dataset


def create_dataloader(dataset, opt, phase):
    '''create dataloader '''
    dataset_opt = opt["datasets"][phase]
    distributed = opt.get('distributed', False)

    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=opt['local_rank'],
            shuffle=(phase == 'train'))
    else:
        sampler = None

    if phase == 'train':

        return DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'],
            shuffle= False if distributed else dataset_opt['use_shuffle'],
            num_workers=dataset_opt['num_workers'],
            pin_memory=True,
            sampler=sampler)
    elif phase == 'val':
        return DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'],
            shuffle=False,
            num_workers=dataset_opt['num_workers'],
            pin_memory=True,
            sampler=sampler)
    else:
        raise NotImplementedError(
            'Dataloader [{:s}] is not found.'.format(phase))



def create_dataset(opt, phase, stage):
    '''create dataset'''
    dataset_opt = opt['datasets'][phase]

    if stage == 1:
        dataset = S1Dataset(dataroot=dataset_opt['dataroot'],
                            phase=dataset_opt['phase'],
                            val_volume_idx=dataset_opt.get('val_volume_idx', None),
                            val_frame_idx=dataset_opt.get('val_frame_idx', None),
                            val_slice_idx=dataset_opt.get('val_slice_idx', None),
                            gt_path=opt['datasets']['val']['gt_dataroot'],
                            norm_method=dataset_opt['norm_method'],
                            time_seq=dataset_opt['time_seq'],
                            use_mmap=dataset_opt['use_mmap'],
                            chunk_size=dataset_opt['chunk_size'],
                            window_size=dataset_opt['window_size'])

    elif stage == 2:
        dataset = S2Dataset(dataroot=dataset_opt['dataroot'],
                            phase=dataset_opt['phase'],
                            val_volume_idx=dataset_opt.get('val_volume_idx', None),
                            val_frame_idx=dataset_opt.get('val_frame_idx', None),
                            val_slice_idx=dataset_opt.get('val_slice_idx', None),
                            gt_path=opt['datasets']['val']['gt_dataroot'],
                            predenoise_path = opt['predenoise_path'],
                            norm_method=dataset_opt['norm_method'],
                            time_seq=dataset_opt['time_seq'],
                            use_mmap=dataset_opt['use_mmap'],
                            chunk_size=dataset_opt['chunk_size'],
                            window_size=dataset_opt['window_size'],
                            match_file=opt['match_file'],
                            global_cond=dataset_opt['global_cond'],
                            time_cond=dataset_opt['time_cond'],
                            space_cond=dataset_opt['space_cond'],
                            )
    else:
        raise ValueError(f"Invalid stage: {stage}. Expected 1 or 2 ")


    logger = logging.getLogger(phase)
    logger.info('dataset [{:s}] is created.'.format(dataset_opt['name']))

    return dataset
