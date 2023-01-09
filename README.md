# YOLOv3

re-implementation of yolo v3
Please refer to https://arxiv.org/abs/1506.01497 
 
### Dataset
- [x] COCO


### Data Augmentation
```
DETR augmentation 
Random Photo Distortion
Random zoom out
Random crop
```

### Training Setting

```
- batch size : 64
- optimizer : SGD
- epoch : 273 
- initial learning rate 0.001 (warm up 4000)
- burn-in 
- weight decay : 5e-4
- momentum : 0.9
- scheduler : step LR (min : 5e-5)
```


|methods        | Traning Dataset        |    Testing Dataset     | Resolution | AP      |AP50   |AP75    | Time | Fps  |
|---------------|------------------------| ---------------------- | ---------- | ------- |-------|--------|:----:| ---- |
|papers         | COCOtrain2017          |  COCO test-dev         | 416 x 416  |  31.0   |55.3   |34.4    |29    |34.48 |
|our repo       | COCOtrain2017          |  COCOval2017(minival)  | 416 x 416  |**32.5** |54.3   |34.1    |**26.45**|**37.81**|

264 epoch 기준

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.325
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.543
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.341
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.165
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.347
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.473
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.275
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.411
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.426
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.242
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.444
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.593
```

### Scheduler

- we use step LR scheduler and learning rate warmup(burning) scheme 
1833 x 218 = 399594
1833 x 246 = 450918
1833 x 273 = 500409

### burn-in

- batch 64 & iteration 4000
- lr : 0 to 1e-3 (i/4000)

### training

- batch : 64
- scheduler : step LR
- loss : sse + bce
- dataset : coco
- epoch : 273
