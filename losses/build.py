from losses.loss import YoloV3Loss


def build_loss(opts):
    criterion = YoloV3Loss(opts)
    return criterion
