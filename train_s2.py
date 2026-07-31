import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
from tqdm import tqdm
import tifffile as tiff
import copy

import torch.distributed as dist

import data as Data
import argparse
import logging
import utils.logger as Logger
import utils.metrics as Metrics
from model.adaptive_inversion import adapt_step
import model as Model


def setup_logger(opt):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # Logger.seed_everything(123 + opt['local_rank'])

    if opt['local_rank'] == 0:
        Logger.setup_logger('train', opt['path']['log'], 'train', level=logging.INFO, screen=True)
        Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO, screen=True)
        logger = logging.getLogger(opt['phase'])
        logger.info(Logger.dict2str(opt))

def init_dataloader(opt, phase):
    dataset = Data.create_dataset(opt, phase, stage=2)
    dataloader = Data.create_dataloader(dataset, opt, phase)
    return dataloader

# def adaptive_step(opt, phase, custom_s1=None):    # fixme: 程序先创建空的 adaptive_matching.txt，adaptive_step() 本来准备计算 t∗并写入该文件，但创建数据集时，S2Dataset 提前读取这个空文件导致直接报错
#     opt["datasets"]['val']['batch_size'] //= 2
#     opt["datasets"]['train']['batch_size'] //= 2
#     logger = logging.getLogger(phase)
#     opt = Logger.val_set(opt)
#     match_file = opt['match_file']
#
#     if not os.path.getsize(match_file):
#         dataset = Data.create_dataset(opt, phase=phase, stage=2)
#         dataloader = Data.create_dataloader(dataset, opt, 'val')
#         adapt_step(dataloader, opt, custom_s1=custom_s1)
#         logger.info('Match state done, saved in {}'.format(match_file))
#     else:
#         logger.info('Match state file exists: {}'.format(match_file))

def adaptive_step(opt, phase, custom_s1=None):
    opt["datasets"]["val"]["batch_size"] = max(
        1, opt["datasets"]["val"]["batch_size"] // 2
    )
    opt["datasets"]["train"]["batch_size"] = max(
        1, opt["datasets"]["train"]["batch_size"] // 2
    )

    logger = logging.getLogger(phase)
    opt = Logger.val_set(opt)
    match_file = opt["match_file"]

    if not os.path.isfile(match_file) or os.path.getsize(match_file) == 0:
        # 生成 t* 时，数据集不能读取尚为空的匹配文件
        matching_opt = copy.deepcopy(opt)
        matching_opt["match_file"] = None

        dataset = Data.create_dataset(
            matching_opt,
            phase=phase,
            stage=2
        )
        dataloader = Data.create_dataloader(
            dataset,
            matching_opt,
            phase
        )

        # adapt_step 仍使用原 opt，将结果写入 match_file
        adapt_step(
            dataloader,
            opt,
            custom_s1=custom_s1
        )

        logger.info(
            "Match state done, saved in {}".format(match_file)
        )
    else:
        logger.info(
            "Match state file exists: {}".format(match_file)
        )


def train_model(opt, train_loader, val_loader, trainer):
    current_step = trainer.begin_step
    current_epoch = trainer.begin_epoch
    n_iter = opt['train']['n_iter']
    logger = logging.getLogger('train')

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))
    logger.info('train start')

    trainer.init_noise_schedule(opt['model']['beta_schedule'])

    while current_step < n_iter:
        current_epoch += 1

        if opt['distributed']:
            train_loader.sampler.set_epoch(current_epoch)

        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > n_iter:
                break
            trainer.feed_data(train_data)
            trainer.optimize_parameters()


            # log
            if opt['local_rank'] == 0 and current_step % opt['train']['print_freq'] == 0:
                logs = trainer.get_current_log()
                message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                    current_epoch, current_step)
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                logger.info(message)


            # test
            results_dir = opt['path']['results']
            save_denoised_path = os.path.join(results_dir, f'S2_I{current_step:05d}_E{current_epoch:03d}_denoised.tif')
            save_input_path = os.path.join(results_dir, 'input.tif')
            save_gt_path = os.path.join(results_dir, 'gt.tif') if opt['exist_gt'] else None

            if current_step % opt['train']['val_freq'] == 0:

                denoised_imgs, input_imgs, gt_imgs = gen_validate_results(trainer, val_loader, opt)

                Metrics.save_img(denoised_imgs, save_denoised_path)
                if current_step == opt['train']['val_freq']:
                    Metrics.save_img(input_imgs, save_input_path)
                    if opt['exist_gt']:
                        Metrics.save_img(gt_imgs, save_gt_path)

                if opt['exist_gt']:
                    evaluator = Metrics.MetricEvaluator()
                    batch_metrics = evaluator.evaluate_batch(gt_imgs, denoised_imgs, input_imgs)

                    evaluator.cache['denoised_psnr'].extend(batch_metrics['denoised_psnr'])
                    evaluator.cache['denoised_ssim'].extend(batch_metrics['denoised_ssim'])
                    evaluator.cache['input_psnr'].extend(batch_metrics['input_psnr'])
                    evaluator.cache['input_ssim'].extend(batch_metrics['input_ssim'])

                    evaluate_results = evaluator.aggregate_results()

                    logger.info("Evaluation metrics: %s", evaluate_results)


            if opt['local_rank'] == 0 and current_step % opt['train']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                trainer.save_network(current_epoch, current_step, save_last_only=False)


    logger.info('End of training.')

def gen_validate_results(trainer, val_loader, opt):
    denoised_list, input_list, gt_list = [], [], []
    for val_data in tqdm(val_loader, desc='test', disable=opt['local_rank']!=0):
        trainer.feed_data(val_data)
        trainer.test(continous=True)
        visuals = trainer.get_current_visuals()

        denoised_list.append(visuals['denoised'])
        input_list.append(visuals['Y']) # TODO: need change

        if opt["exist_gt"]:
            gt_list.append(val_data['gt'].cpu())

    denoised_imgs = torch.cat(denoised_list, dim=0)
    input_imgs = torch.cat(input_list, dim=0)
    gt_imgs = torch.cat(gt_list, dim=0) if gt_list else None

    if opt['distributed']:
        denoised_imgs = gather_tensor(denoised_imgs)
        input_imgs = gather_tensor(input_imgs)
        gt_imgs = gather_tensor(gt_imgs) if gt_imgs is not None else None

    if opt['local_rank'] == 0:
        denoised_imgs = Metrics.tensor2img(denoised_imgs)
        input_imgs = Metrics.tensor2img(input_imgs)
        gt_imgs = Metrics.tensor2img(gt_imgs) if gt_imgs is not None else None

    return denoised_imgs, input_imgs, gt_imgs

def gather_tensor(tensor):
    if not dist.is_initialized():
        return tensor
    world_size = dist.get_world_size()
    if world_size == 1:
        return tensor

    output_tensors = [torch.zeros_like(tensor).to(tensor.device) for _ in range(world_size)]
    dist.all_gather(output_tensors, tensor)
    return torch.cat(output_tensors, dim=0)


def validate_model(opt, val_loader, trainer):
    assert opt['path']['resume_state'] is not None

    # set up save path
    dir_basename = os.path.basename(opt["datasets"]['val']['dataroot'])
    save_dir = os.path.join(opt['path']['results'], f'validation_{dir_basename}')
    os.makedirs(save_dir, exist_ok=True)

    denoised_save_path = os.path.join(save_dir, f'S2_{trainer.ckpt_name}_denoised.tif')
    input_save_path = os.path.join(save_dir, 'input.tif')
    gt_save_path = os.path.join(save_dir, 'gt.tif') if opt['exist_gt'] else None

    if opt['local_rank'] == 0:
        denoised_writer = tiff.TiffWriter(denoised_save_path, bigtiff=True)
        input_writer = tiff.TiffWriter(input_save_path, bigtiff=True)
        gt_writer = tiff.TiffWriter(gt_save_path, bigtiff=True) if opt['exist_gt'] else None
    evaluator = Metrics.MetricEvaluator()

    for val_data in tqdm(val_loader, desc="Validation", disable=opt['local_rank']!=0):
        trainer.feed_data(val_data)
        trainer.test(continous=True)
        visuals = trainer.get_current_visuals()

        denoised = gather_tensor(visuals['denoised'])
        input = gather_tensor(visuals['Y'])
        gt = gather_tensor(val_data['gt'].cpu()) if opt['exist_gt'] else None

        if opt['local_rank'] == 0:
            denoised_img = Metrics.tensor2img(denoised)
            input_img = Metrics.tensor2img(input)
            gt_img = Metrics.tensor2img(gt) if opt['exist_gt'] else None

            denoised_writer.write(denoised_img, photometric='minisblack')
            input_writer.write(input_img, photometric='minisblack')

            if opt['exist_gt']:
                gt_writer.write(gt_img, photometric='minisblack')

                batch_metrics = evaluator.evaluate_batch(gt_img, denoised_img, input_img)

                evaluator.cache['denoised_psnr'].extend(batch_metrics['denoised_psnr'])
                evaluator.cache['denoised_ssim'].extend(batch_metrics['denoised_ssim'])
                evaluator.cache['input_psnr'].extend(batch_metrics['input_psnr'])
                evaluator.cache['input_ssim'].extend(batch_metrics['input_ssim'])

                evalu_results = evaluator.aggregate_results()

    logger = logging.getLogger('val')

    if opt['local_rank'] == 0:
        denoised_writer.close()
        input_writer.close()

        if gt_writer:
            gt_writer.close()
            logger.info("Evaluation metrics: %s", evalu_results)
            evaluator.save_csv(save_dir, val_loader.dataset.images_shape)
    logger.info('End of validation. results saved in {}'.format(save_dir))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/demo_config.json',
                        help='JSON configuration file')
    parser.add_argument('-p', '--phase', type=str, choices=['val'],
                        help='Run inference', default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default='0')
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-name', type=str, default='dual_demo')



    args = parser.parse_args()
    Logger.ddp_setup(args.gpu_ids)
    opt = Logger.parse(args, stage=2)
    setup_logger(opt)

    # required_keys: ['denoiser', 'opt_s1', 'scheduler'] if custom_s1 is not None
    adaptive_step(copy.deepcopy(opt), args.phase, custom_s1=None)   # NOTE: 启动ADI，在 Stage 2 前计算每张图像的 t*

    # stage 1 model: predenoiser
    trainer = Model.create_model(opt, stage=2, custom_s1=None)


    # train or validation
    if args.phase == 'train':
        train_loader = init_dataloader(opt, 'train')
        val_loader = init_dataloader(opt, 'val')

        train_model(opt, train_loader, val_loader, trainer)

    elif args.phase == 'val':
        opt = Logger.val_set(opt)
        val_loader = init_dataloader(opt, 'val')
        validate_model(opt, val_loader, trainer)

if __name__ == "__main__":
    main()
