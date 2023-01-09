import time
import os
import torch
from tqdm import tqdm


def train_one_epoch(epoch, device, vis, loader, model, criterion, optimizer, scheduler, opts):

    print('Training of epoch [{}]'.format(epoch))
    tic = time.time()
    model.train()
    model.module.anchor.to_device(device)

    for idx, data in enumerate(tqdm(loader)):

        # burn in process
        if opts.burn_in is not None:
            burn_in_idx = idx + epoch * len(loader)
            if burn_in_idx < opts.burn_in:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = opts.lr * ((burn_in_idx + 1) / opts.burn_in) ** 4

        images = data[0]
        boxes = data[1]
        labels = data[2]

        images = images.to(device)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        pred = model(images)
        loss, losses, _ = criterion(pred, boxes, labels, model.module.anchor)

        # sgd
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        toc = time.time()

        for param_group in optimizer.param_groups:
            lr = param_group['lr']

        # for each steps
        if idx % opts.train_vis_step == 0 or idx == len(loader) - 1:
            print('Epoch: [{0}]\t'
                  'Step: [{1}/{2}]\t'
                  'Loss: {loss:.4f}\t'
                  'Learning rate: {lr:.7f} s \t'
                  'Time : {time:.4f}\t'
                  .format(epoch, idx, len(loader),
                          loss=loss,
                          lr=lr,
                          time=toc - tic))
            if opts.rank == 0:
                if vis is not None:
                    # loss plot
                    vis.line(X=torch.ones((1, 6)).cpu() * idx + epoch * loader.__len__(),  # step
                             Y=torch.Tensor([loss, losses[0], losses[1], losses[2], losses[3], losses[4]])
                             .unsqueeze(0).cpu(),
                             win='train_loss_' + opts.name,
                             update='append',
                             opts=dict(xlabel='step',
                                       ylabel='Loss',
                                       title='train_loss_' + opts.name,
                                       legend=['Total Loss', 'xy Loss', 'wh Loss', 'obj Loss', 'no obj Loss', 'cls Loss']))

    # # 각 epoch 마다 저장
    if not os.path.exists(opts.save_dir):
        os.mkdir(opts.save_dir)

    checkpoint = {'epoch': epoch,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'scheduler_state_dict': scheduler.state_dict()}

    if epoch % opts.save_epoch == 0:
        torch.save(checkpoint, os.path.join(opts.save_dir, opts.name, opts.name + '.{}.pth.tar'.format(epoch)))
        print('save pth file of {} epoch!'.format(epoch))


