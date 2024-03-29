import torch
from models.model import YoloV3, Darknet53
from torch.nn.parallel import DistributedDataParallel as DDP


def build_model(opts):
    if opts.distributed:
        model = YoloV3(baseline=Darknet53(pretrained=opts.pretrained), num_classes=opts.num_classes)
        model = model.cuda(int(opts.gpu_ids[opts.rank]))
        model = DDP(module=model,
                    device_ids=[int(opts.gpu_ids[opts.rank])],
                    find_unused_parameters=True)
    else:
        # IF DP
        model = YoloV3(baseline=Darknet53(pretrained=opts.pretrained), num_classes=opts.num_classes)
        model = torch.nn.DataParallel(module=model, device_ids=[int(id) for id in opts.gpu_ids])
    return model


