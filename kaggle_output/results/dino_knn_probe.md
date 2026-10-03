k-NN (k=20) multi-label scene classification on VOC2007 test (4952 images), memory bank = 4000 random VOC train images.

| Global embedding | dim | mAP |
|---|---|---|
| DINOv2 ViT-S/14 CLS (self-supervised, frozen) | 384 | 0.8900 |
| YOLOv8n SPPF/P5 avg-pool (COCO-supervised) | 256 | 0.8167 |

Per-class AP:

| class | DINOv2 ViT-S/14 CLS | YOLOv8n SPPF/P5 avg-pool |
|---|---|---|
| aeroplane | 0.997 | 0.969 |
| bicycle | 0.944 | 0.849 |
| bird | 0.966 | 0.845 |
| boat | 0.961 | 0.899 |
| bottle | 0.662 | 0.481 |
| bus | 0.935 | 0.827 |
| car | 0.916 | 0.905 |
| cat | 0.962 | 0.871 |
| chair | 0.662 | 0.678 |
| cow | 0.982 | 0.797 |
| diningtable | 0.852 | 0.835 |
| dog | 0.962 | 0.776 |
| horse | 0.981 | 0.943 |
| motorbike | 0.952 | 0.857 |
| person | 0.913 | 0.939 |
| pottedplant | 0.577 | 0.624 |
| sheep | 0.995 | 0.827 |
| sofa | 0.765 | 0.740 |
| train | 0.995 | 0.961 |
| tvmonitor | 0.821 | 0.710 |
