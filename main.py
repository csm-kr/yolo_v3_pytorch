import os
import torch
import visdom
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

# data
from datasets.build import build_dataloader
# model
from models.build import build_model
# loss
from losses.build import build_loss
# train
from train import train_one_epoch
# test
from test import test_and_eval
# scheduler
from torch.optim.lr_scheduler import MultiStepLR
# log
from log import XLLogSaver
# util
from utils.util import init_for_distributed, resume
# multi processing
import torch.multiprocessing as mp



def main_worker(rank, opts):

    # 1. opts
    print(opts)

    # 2. distributed
    if opts.distributed:
        init_for_distributed(rank, opts)

    # 3. device
    device = torch.device('cuda:{}'.format(int(opts.gpu_ids[opts.rank])))

    # 4. visdom
    vis = visdom.Visdom(port=opts.visdom_port)

    # 5. data
    train_loader, test_loader = build_dataloader(opts)

    # 6. model
    model = build_model(opts)

    # 7. loss
    criterion = build_loss(opts)

    # 8. optimizer
    optimizer = torch.optim.SGD(params=model.parameters(),
                                lr=opts.lr,
                                momentum=0.9,
                                weight_decay=5e-4)

    # 9. scheduler
    scheduler = MultiStepLR(optimizer=optimizer, milestones=[218, 246], gamma=0.1)

    # 10. log
    xl_log_saver = XLLogSaver(xl_folder_name=os.path.join(opts.save_dir, opts.name),
                              xl_file_name=opts.name,
                              tabs=('epoch', 'mAP', 'val_loss'))

    # 11. resume
    resume(opts, model, optimizer, scheduler)

    # 12. set best result
    result_best = {'epoch': 0, 'mAP': 0., 'val_loss': 0.}

    # for statement
    for epoch in range(opts.start_epoch, opts.epoch):

        if opts.distributed:
            train_loader.sampler.set_epoch(epoch)

        # 13. train
        train_one_epoch(opts=opts,
                        epoch=epoch,
                        device=device,
                        vis=vis,
                        loader=train_loader,
                        model=model,
                        criterion=criterion,
                        optimizer=optimizer,
                        scheduler=scheduler)

        # 14. test
        test_and_eval(opts=opts,
                      epoch=epoch,
                      device=device,
                      vis=vis,
                      loader=test_loader,
                      model=model,
                      criterion=criterion,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      xl_log_saver=xl_log_saver,
                      result_best=result_best,
                      is_load=False)

        scheduler.step()


if __name__ == "__main__":
    import configargparse
    from config import get_args_parser
    parser = configargparse.ArgumentParser('YOLOv3 train', parents=[get_args_parser()])
    opts = parser.parse_args()

    if len(opts.gpu_ids) > 1:
        opts.distributed = True

    opts.world_size = len(opts.gpu_ids)
    opts.num_workers = len(opts.gpu_ids) * 4

    if opts.distributed:
        mp.spawn(main_worker,
                 args=(opts,),
                 nprocs=opts.world_size,
                 join=True)
    else:
        main_worker(opts.rank, opts)




