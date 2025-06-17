wandb online
# wandb offline

python train_simple.py \
  --img 640 \
  --batch-size 128 \
  --epochs 50 \
  --data data/kaist-rgbt.yaml \
  --cfg models/yolov5n_kaist-rgbt-anchor-v2.yaml \
  --device 0 \
  --weights yolov5n.pt \
  --workers 8 \
  --name yolov5n-rgbt-enhanced-full-aug-anchor-v2-multi-class \
  --rgbt \
  --augment \
  --mixup 0.40

## Large Models

# python train_simple.py \
#   --img 640 \
#   --batch-size 24 \
#   --epochs 50 \
#   --data data/kaist-rgbt.yaml \
#   --cfg models/yolov5n_kaist-rgbt-large-anchor-v2.yaml \
#   --device 2 \
#   --weights yolov5n.pt \
#   --workers 8 \
#   --name yolov5n-rgbt-large-enhanced-full-aug-anchor-v2-multi-class \
#   --rgbt \
#   --augment \
#   --mixup 0.40

# python train_simple.py \
#   --img 640 \
#   --batch-size 32 \
#   --epochs 100 \
#   --data data/kaist-rgbt.yaml \
#   --cfg models/yolov5n_kaist-rgbt-xlarge-anchor-v2.yaml \
#   --device 0 \
#   --weights yolov5n.pt \
#   --workers 8 \
#   --name yolov5n-rgbt-xlarge-enhanced-full-aug-anchor-v2-multi-class \
#   --rgbt \
#   --augment \
#   --mixup 0.40