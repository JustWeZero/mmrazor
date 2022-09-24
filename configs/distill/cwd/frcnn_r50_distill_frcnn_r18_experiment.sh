# 对FPN的4层做蒸馏，teacher和student的输出通道数都为256
# 原训练模型精度为0.243，蒸馏完成的模型精度为0.2450，涨了0.2的点数，好像没什么用
export GPU=4 && LR=0.02 && CONFIG="cwd_fpn_lvl_frcnn_r50_lka_fpn_frcnn_r18_lka_fpn_1x_VisDrone640"
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 tools/mmdet/dist_train.sh configs/distill/cwd/${CONFIG}.py ${GPU} \
 --work-dir ../VisDrone_cache/lka_fpn/${CONFIG}/slice_640x640_lr${LR}_1x_${GPU}g/ \
 --cfg-options optimizer.lr=${LR} data.samples_per_gpu=4 data.workers_per_gpu=2

# 先对FPN的4层做蒸馏，teacher和student的输出通道数都为64，
# 原训练模型精度为0.2540，r50_256的精度为0.2670
# Epoch(val) [12][137]   bbox_mAP: 0.2130, bbox_mAP_50: 0.4360, bbox_mAP_75: 0.1860, bbox_mAP_s: 0.1370, bbox_mAP_m: 0.3080, bbox_mAP_l: 0.3670,
# bbox_mAP_copypaste: 0.213 0.436 0.186 0.137 0.308 0.367
# 精度有恢复，但是还是不够，看看加入对AEM的注意力的蒸馏可不可以
export GPU=4 && LR=0.08 && CONFIG="cwd_fpnlvl_64ch_frcnn_r50_lka_fpn_frcnn_r18_lka_fpn_1x_VisDrone640"
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 tools/mmdet/dist_train.sh configs/distill/cwd/${CONFIG}.py ${GPU} \
 --work-dir ../VisDrone_cache/lka_fpn/${CONFIG}/slice_640x640_lr${LR}_1x_${GPU}g/ \
 --cfg-options optimizer.lr=${LR} data.samples_per_gpu=8 data.workers_per_gpu=2

# 训到第7个epoch，att的distill_loss变成nan了，很难不说是注意力的问题……加了注意力就变nan了
export GPU=4 && LR=0.08 && CONFIG="cwd_fpnlvl_aematt_frcnn_r50_lka_fpn_frcnn_r18_lka_fpn_1x_VisDrone640"
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 tools/mmdet/dist_train.sh configs/distill/cwd/${CONFIG}.py ${GPU} \
 --work-dir ../VisDrone_cache/lka_fpn/${CONFIG}/slice_640x640_lr${LR}_1x_${GPU}g/ \
 --cfg-options optimizer.lr=${LR} data.samples_per_gpu=8 data.workers_per_gpu=2

# 试试看减小2倍的学习率看看什么结果
# Epoch(val) [12][137]   bbox_mAP: 0.1880, bbox_mAP_50: 0.3990, bbox_mAP_75: 0.1590, bbox_mAP_s: 0.1230, bbox_mAP_m: 0.2720, bbox_mAP_l: 0.3270,
# bbox_mAP_copypaste: 0.188 0.399 0.159 0.123 0.272 0.327
# 虽然有些违反了线性缩放原则，但是大差不差了，只有2倍而已。精度比起单FPN掉了接近3个点还挺多，
export GPU=4 && LR=0.04 && CONFIG="cwd_fpnlvl_aematt_frcnn_r50_lka_fpn_frcnn_r18_lka_fpn_1x_VisDrone640"
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 tools/mmdet/dist_train.sh configs/distill/cwd/${CONFIG}.py ${GPU} \
 --work-dir ../VisDrone_cache/lka_fpn/${CONFIG}/slice_640x640_lr${LR}_1x_${GPU}g/ \
 --cfg-options optimizer.lr=${LR} data.samples_per_gpu=8 data.workers_per_gpu=2

# 添加蒸backbone输出后到FPN降维的那个conv，直接蒸backbone的卷积层的话通道数不同做不了loss
# Epoch(val) [12][137]   bbox_mAP: 0.2060, bbox_mAP_50: 0.4250, bbox_mAP_75: 0.1800, bbox_mAP_s: 0.1340, bbox_mAP_m: 0.2980, bbox_mAP_l: 0.3370,
# bbox_mAP_copypaste: 0.206 0.425 0.180 0.134 0.298 0.337
# Epoch(val) [11][137]   bbox_mAP: 0.2060, bbox_mAP_50: 0.4180, bbox_mAP_75: 0.1800, bbox_mAP_s: 0.1330, bbox_mAP_m: 0.2970, bbox_mAP_l: 0.3360,
# bbox_mAP_copypaste: 0.206 0.418 0.180 0.133 0.297 0.336
# 貌似没太大作用，没有单蒸FPN的效果好
export GPU=4 && LR=0.08 && CONFIG="cwd_bbout_fpnlvl_64ch_frcnn_r50_lka_fpn_frcnn_r18_lka_fpn_1x_VisDrone640"
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 tools/mmdet/dist_train.sh configs/distill/cwd/${CONFIG}.py ${GPU} \
 --work-dir ../VisDrone_cache/lka_fpn/${CONFIG}/slice_640x640_lr${LR}_1x_${GPU}g/ \
 --cfg-options optimizer.lr=${LR} data.samples_per_gpu=8 data.workers_per_gpu=2