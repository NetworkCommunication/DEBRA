import os
import psutil
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import numpy as np
from tqdm import tqdm
from arguments import get_args
from data_transform import get_trans
from networks import get_model
from tools import AverageMeter, knn_monitor, Logger, file_exist_check
from dataset import get_dataset
from optimizers import get_optimizer, LR_Scheduler
from linear_training import main as linear_eval
from datetime import datetime
import time

def main(device, args):
    start_time = time.time()
    train_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(transform=get_trans(train=True, **args.aug_kwargs),
                            train=True, **args.dataset_kwargs, dir=args.train.train_data_dir),
        shuffle=True,
        batch_size=args.train.batch_size,
        **args.dataloader_kwargs
    )
    memory_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(
            transform=get_trans(train=False, train_classifier=False, **args.aug_kwargs),
            train=True,
            **args.dataset_kwargs, dir=args.train.train_data_dir),
        shuffle=False,
        batch_size=args.train.batch_size,
        **args.dataloader_kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=get_dataset( 
            transform=get_trans(train=False, train_classifier=False, **args.aug_kwargs),
            train=False,
            **args.dataset_kwargs, dir=args.train.test_data_dir),
        shuffle=False,
        batch_size=args.train.batch_size,
        **args.dataloader_kwargs
    )

    # define model
    model = get_model(args.model).to(device)
    print(model)

    optimizer = get_optimizer(
        args.train.optimizer.name, model,
        lr=args.train.base_lr*args.train.batch_size/128,
        momentum=args.train.optimizer.momentum,
        weight_decay=args.train.optimizer.weight_decay)

    lr_scheduler = LR_Scheduler(
        optimizer,
        args.train.warmup_epochs, args.train.warmup_lr*args.train.batch_size/128,
        args.train.num_epochs, args.train.base_lr*args.train.batch_size/128, args.train.final_lr*args.train.batch_size/128,
        len(train_loader),
        constant_predictor_lr=True  # 预测器学习率保持恒定
    )

    logger = Logger(tensorboard=args.logger.tensorboard, matplotlib=args.logger.matplotlib, log_dir=args.log_dir)
    accuracy = 0
    train_log = []
    loss = []
    # Start training
    global_progress = tqdm(range(0, args.train.stop_at_epoch), desc=f'Training')
    # start_time = time.time()
    max_acc = 94.0
    for epoch in global_progress:
        model.train()

        # 获取当前进程的内存使用情况
        # process = psutil.Process(os.getpid())
        # memory_info = process.memory_info()
        # print(f"当前进程使用的内存: {memory_info.rss / (1024 ** 2):.2f} MB")  # rss: 常驻内存集
        # # 获取当前 GPU 的内存使用情况
        # gpu_memory = torch.cuda.memory_allocated()
        # print(f"当前 GPU 内存使用: {gpu_memory / (1024 ** 2):.2f} MB")

        local_progress=tqdm(train_loader, desc=f'Epoch {epoch}/{args.train.num_epochs}', position=0, disable=args.hide_progress)
        accum_iter = 8
        # start_time = time.time()
        for idx, ((images1, images2), labels) in enumerate(local_progress):

            model.zero_grad()

            data_dict = model.forward(images1.to(device, non_blocking=True), images2.to(device, non_blocking=True))

            loss = data_dict['loss'].mean() # ddp
           # loss = loss / accum_iter
            loss.backward()
            if idx % accum_iter == 0 or idx + 1 == len(local_progress):
                # 平均累积的梯度
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad /= accum_iter  # 对梯度进行平均
                optimizer.step()
                model.zero_grad()

            lr_scheduler.step()
            data_dict.update({'lr':lr_scheduler.get_lr()})

            local_progress.set_postfix(data_dict)
            logger.update_scalers(data_dict)
            # print(torch.cuda.memory_cached())

        if args.train.knn_monitor and epoch % args.train.knn_interval == 0:
            # accuracy = knn_monitor(model.module.backbone, memory_loader, test_loader, device, k=min(args.train.knn_k, len(memory_loader.dataset)), hide_progress=args.hide_progress)
            accuracy = knn_monitor(model.backbone, memory_loader, test_loader, device,
                                   k=min(args.train.knn_k, len(memory_loader.dataset)),
                                   hide_progress=args.hide_progress)
        if accuracy > max_acc:
            model_path = os.path.join(args.ckpt_dir,
                                      f"GA_pre_acc{accuracy:.1f}" + args.dataset.name + '_lr' + str(
                                          args.train.base_lr) +
                                      '_batch_size' + str(args.train.batch_size) + '_imgSize' + str(
                                          args.dataset.image_size) + '.pth')
            torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict()}, model_path)
            max_acc = accuracy

        epoch_dict = {"epoch":epoch, "accuracy":accuracy}
        global_progress.set_postfix(epoch_dict)
        logger.update_scalers(epoch_dict)
        this_time = time.time()
        train_log.append({
            'epoch': epoch,
            'loss':  f"{loss.item():.2f}",
            'accuracy': f"{accuracy:.2f}",
            'lr': lr_scheduler.get_lr(),
            'use_time': this_time
        })
    # Save log
    all_time = (time.time() - start_time)/60
    print(f"Total training time: {all_time:.2f} mins")

    log_dir = 'logs/train_feature'
    os.makedirs(log_dir, exist_ok=True)  # exist_ok=True 可以防止目录已经存在时抛出异常
    file_path = os.path.join(log_dir, f"GA_pre_acc{accuracy:.1f}_all_time{all_time:.2f}mins"+args.dataset.name + '_lr' + str(args.train.base_lr) +
                             '_batch_size' + str(args.train.batch_size) + '_imgSize'+str(args.dataset.image_size) +str(args.model.backbone)+'_log.csv')
    # 保存训练日志到 CSV 文件
    train_df = pd.DataFrame(train_log)
    train_df.to_csv(file_path, index=False)

    # Save checkpoint
    model_path = os.path.join(args.ckpt_dir, f"GA_pre_acc{accuracy:.1f}_all_time{all_time:.2f}mins"+args.dataset.name + '_lr' + str(args.train.base_lr) +
                             '_batch_size' + str(args.train.batch_size) + '_imgSize'+str(args.dataset.image_size) + str(args.model.backbone)+ '.pth')
    torch.save({
        'epoch': epoch+1,
        'state_dict':model.state_dict()
    }, model_path)
    print(f"Model saved to {model_path}")
    # with open(os.path.join(args.log_dir, f"checkpoint_path.txt"), 'w+') as f:
    #     f.write(f'{model_path}')

    if args.linear_eval:
        args.eval_from = model_path
        linear_eval(args)


if __name__ == "__main__":
    args = get_args()

    main(device=args.device, args=args)

    completed_log_dir = args.log_dir.replace('in-progress', 'debug' if args.debug else 'completed')
    # os.rename(args.log_dir, completed_log_dir)

    print(f'Log file has been saved to {completed_log_dir}')

