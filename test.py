import os
import time
import torch
from tqdm import tqdm
from evaluation.evaluator import Evaluator

# for test
from losses.build import build_loss
from models.build import build_model
from datasets.build import build_dataloader


@ torch.no_grad()
def test_and_eval(opts, epoch, device, vis, loader, model, criterion, optimizer=None, scheduler=None,
         xl_log_saver=None, result_best=None, is_load=False):

    if opts.rank == 0:
        print('Validation of epoch [{}]'.format(epoch))
        # ---------- eval ----------
        evaluator = Evaluator(opts=opts)

        # ---------- load ----------
        checkpoint = None
        if is_load:
            checkpoint = torch.load(
                f=os.path.join(opts.save_dir, opts.name, opts.name + '.{}.pth.tar'.format(epoch)),
                map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print('load pth file of {} epoch!'.format(epoch))

        model.eval()
        sum_loss = 0
        tic = time.time()
        model.module.anchor.to_device(device)

        for idx, data in enumerate(tqdm(loader)):

            images = data[0]
            boxes = data[1]
            labels = data[2]

            # ---------- cuda ----------
            images = images.to(device)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            # ---------- loss ----------
            pred = model(images)
            loss, _, _ = criterion(pred, boxes, labels, model.module.anchor)

            sum_loss += loss.item()

            # ---------- eval ----------
            pred_boxes, pred_labels, pred_scores = model.module.predict(preds=pred,
                                                                        list_center_anchors=model.module.anchor.center_anchors,
                                                                        opts=opts)

            if opts.data_type == 'voc':
                info = data[3][0]
                info = (pred_boxes, pred_labels, pred_scores, info['name'], info['original_wh'])

            elif opts.data_type == 'coco':
                img_id = loader.dataset.img_id[idx]
                img_info = loader.dataset.coco.loadImgs(ids=img_id)[0]
                coco_ids = loader.dataset.coco_ids
                info = (pred_boxes, pred_labels, pred_scores, img_id, img_info, coco_ids)

            evaluator.get_info(info)
            toc = time.time()

            # ---------- print ----------
            if idx % opts.test_vis_step == 0 or idx == len(loader) - 1:
                print('Epoch: [{0}]\t'
                      'Step: [{1}/{2}]\t'
                      'Loss: {loss:.4f}\t'
                      'Time : {time:.4f}\t'
                      .format(epoch,
                              idx, len(loader),
                              loss=loss,
                              time=toc - tic))

        mAP = evaluator.evaluate(loader.dataset)
        mean_loss = sum_loss / len(loader)
        print(mAP)

        if vis is not None:
            # loss plot
            vis.line(X=torch.ones((1, 2)).cpu() * epoch,  # step
                     Y=torch.Tensor([mean_loss, mAP]).unsqueeze(0).cpu(),
                     win='test_loss_' + opts.name,
                     update='append',
                     opts=dict(xlabel='step',
                               ylabel='test',
                               title='test_loss_' + opts.name,
                               legend=['test Loss', 'mAP']))

        if xl_log_saver is not None:
            xl_log_saver.insert_each_epoch(contents=(epoch, mAP, mean_loss))

        # save best.pth.tar
        if result_best is not None and optimizer is not None and scheduler is not None:
            if result_best['mAP'] < mAP:
                print("update best model from {:.4f} to {:.4f}".format(result_best['mAP'], mAP))
                result_best['epoch'] = epoch
                result_best['mAP'] = mAP
                if checkpoint is None:
                    checkpoint = {'epoch': epoch,
                                  'model_state_dict': model.state_dict(),
                                  'optimizer_state_dict': optimizer.state_dict(),
                                  'scheduler_state_dict': scheduler.state_dict()}
                print('save pth file of best epoch!')
                torch.save(checkpoint, os.path.join(opts.save_dir, opts.name, opts.name + '.best.pth.tar'))

        return


def test_worker(rank, opts):

    # 1. config
    print(opts)

    # 2. device
    device = torch.device('cuda:{}'.format(int(opts.gpu_ids[opts.rank])))

    # 3. visdom
    vis = None

    # 4. data(set/loader)
    _, test_loader = build_dataloader(opts)

    # 5. model
    model = build_model(opts)
    model = model.to(device)

    # 6. loss
    criterion = build_loss(opts)

    test_and_eval(opts=opts,
                  epoch=opts.test_epoch,
                  vis=vis,
                  device=device,
                  loader=test_loader,
                  model=model,
                  criterion=criterion,
                  is_load=True)


if __name__ == '__main__':

    import configargparse
    from config import get_args_parser
    parser = configargparse.ArgumentParser('YOLO v3 test', parents=[get_args_parser()])
    opts = parser.parse_args()

    opts.world_size = len(opts.gpu_ids)
    opts.num_workers = len(opts.gpu_ids) * 4

    print(opts)
    test_worker(0, opts)



