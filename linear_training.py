import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
from tqdm import tqdm
from arguments import get_args
from data_transform import get_trans
from networks import get_model, get_backbone, get_prediction, get_projector
from tools import AverageMeter
from dataset import get_dataset
from optimizers import get_optimizer, LR_Scheduler

def main(args):

    train_loader = torch.utils.data.DataLoader(
        dataset=get_dataset( 
            transform=get_trans(train=False, train_classifier=True, **args.aug_kwargs),
            train=True, 
            **args.dataset_kwargs,
            train_feature=False, dir=args.eval.train_data_dir
        ),
        batch_size=args.eval.batch_size,
        shuffle=True,
        **args.dataloader_kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(
            transform=get_trans(train=False, train_classifier=False, **args.aug_kwargs),
            train=False,
            **args.dataset_kwargs,
            train_feature=False, dir=args.eval.test_data_dir
        ),
        batch_size=args.eval.batch_size,
        shuffle=False,
        **args.dataloader_kwargs,
    )


    model = get_backbone(args.model.backbone)
    # classifier = nn.Linear(in_features=model.output_dim, out_features=args.num_classes, bias=True).to(args.device)
    # projector = get_projector(args.model)
    # model.output_dim = projector.layer3[0].out_features

    classifier = get_prediction(model.output_dim, args.num_classes).to(args.device)
    print(classifier)
    # 加载预训练权重
    args.eval_from = 'checkpoints/ResNet18/pre_acc95.4EuroSAT_lr0.03_batch_size128_imgSize64.pth'
    assert args.eval_from is not None
    save_dict = torch.load(args.eval_from, map_location='cpu')

    # for k in save_dict['state_dict'].keys():
    #     print(k)
    # 仅加载backbone部分的权重
    msg = model.load_state_dict({k[9:]:v for k, v in save_dict['state_dict'].items() if k.startswith('backbone.')}, strict=False)
    # projector.load_state_dict({k[10:]: v for k, v in save_dict['state_dict'].items() if k.startswith('projector.')}, strict=True)
    
    # print(msg)
    model = model.to(args.device)
    # model = torch.nn.DataParallel(model)

    # if torch.cuda.device_count() > 1: classifier = torch.nn.SyncBatchNorm.convert_sync_batchnorm(classifier)
    classifier = torch.nn.DataParallel(classifier)
    # define optimizer
    optimizer = get_optimizer(
        args.eval.optimizer.name, classifier, 
        lr=args.eval.base_lr*args.eval.batch_size/128,
        momentum=args.eval.optimizer.momentum, 
        weight_decay=args.eval.optimizer.weight_decay)

    # define lr scheduler
    lr_scheduler = LR_Scheduler(
        optimizer,
        args.eval.warmup_epochs, args.eval.warmup_lr*args.eval.batch_size/128,
        args.eval.num_epochs, args.eval.base_lr*args.eval.batch_size/128, args.eval.final_lr*args.eval.batch_size/128,
        len(train_loader),
    )

    loss_meter = AverageMeter(name='Loss')
    train_acc_meter = AverageMeter(name='Train_acc')
    acc_meter = AverageMeter(name='Accuracy')
    train_lr = AverageMeter(name='Learning Rate')

    # Start training
    train_log = []
    for epoch in range(args.eval.stop_at_epoch):
        loss_meter.reset()
        train_acc_meter.reset()
        model.eval()
        classifier.train()
        test_progress = tqdm(test_loader, desc=f'Evaluating', position=0, disable=False)
        train_progress = tqdm(train_loader, desc=f'Epoch {epoch}/{args.eval.num_epochs}', position=0, disable=False)
        
        for idx, (images, labels) in enumerate(train_progress):

            classifier.zero_grad()
            model.zero_grad()
            with torch.no_grad():
                feature = model(images.to(args.device))

            preds = classifier(feature)
            loss = F.cross_entropy(preds, labels.to(args.device))

            # 计算准确率
            _, predicted = preds.max(1)
            correct = (predicted == labels.to(args.device)).sum().item()
            accuracy = (correct / labels.size(0))*100

            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())
            train_acc_meter.update(accuracy)
            lr = lr_scheduler.step()
            train_lr.update(lr)
            train_progress.set_postfix({'lr':lr, "loss":loss_meter.val, 'loss_avg':loss_meter.avg,
                                       'train_acc':train_acc_meter.avg})

        classifier.eval()
        # correct, total = 0, 0
        acc_meter.reset()
        for idx, (images, labels) in enumerate(test_progress):
            with torch.no_grad():
                feature = model(images.to(args.device))
                preds = classifier(feature).argmax(dim=1)
                correct = (preds == labels.to(args.device)).sum().item()
                acc_meter.update((correct / preds.shape[0])*100)
            test_progress.set_postfix({'test_acc': acc_meter.avg})

        # 每个epoch结束后保存数据
        train_log.append({
            'epoch': epoch,
            'loss': loss_meter.avg,
            'train_0.2 accuracy': train_acc_meter.avg,
            'lr': train_lr.avg,
            'test accuracy': acc_meter.avg
        })
    log_dir = 'logs/train_linear/ResNet18'

    last_test_acc = train_log[-1]['test accuracy']
    # 获取文件名（去除路径部分）
    file_name = os.path.basename(args.eval_from)
    # 去除扩展名 '.pth'
    file_name = file_name.replace('.pth', '')
    os.makedirs(log_dir, exist_ok=True)  # exist_ok=True 可以防止目录已经存在时抛出异常
    file_path = os.path.join(log_dir, 'acc' + f"{last_test_acc:.1f}"+'_lr' + str(args.eval.base_lr)+"_Pretrain="+file_name+'.csv')

    # 保存训练日志到 CSV 文件
    train_df = pd.DataFrame(train_log)
    train_df.to_csv(file_path, index=False)


if __name__ == "__main__":
    main(args=get_args())
















