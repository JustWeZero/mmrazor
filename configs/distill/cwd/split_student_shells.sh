export GPU=4 && LR=0.08 && CONFIG="faster_rcnn_r18_lka_fpn_1x_VisDrone640"
python tools/model_converters/split_distill_checkpoint.py \
   ../mmdetection/configs2/VisDrone/lka_fpn/${CONFIG}.py \
   ../VisDrone_cache/lka_fpn/cwd_fpnlvl_64ch_frcnn_r50_lka_fpn_frcnn_r18_lka_fpn_1x_VisDrone640/slice_640x640_lr${LR}_1x_${GPU}g/best_bbox_mAP_epoch_12.pth \
   --output_path ../VisDrone_cache/lka_fpn/${CONFIG}/slice_640x640_lr${LR}_1x_${GPU}g/best_bbox_mAP_epoch_12_student.pth
