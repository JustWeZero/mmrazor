import os.path as osp
root_dir='/home/group5/lzj/VisDrone_cache'
algorithm = 'lka_fpn'
# config = 'faster_rcnn_r50_lka_fpn_outch64_1x_VisDrone640'
config = "cwd_fpnlvl_64ch_frcnn_r50_lka_fpn_frcnn_r18_lka_fpn_1x_VisDrone640"
wh_lr_schdule_gpus = 'slice_640x640_lr0.08_1x_4g'
pth_file_name = 'best_bbox_mAP_epoch_12_student.pth'

total_pth_path = osp.join(root_dir,algorithm,config,wh_lr_schdule_gpus,pth_file_name)

import torch  # 命令行是逐行立即执行的
content = torch.load(total_pth_path)
# print(content.keys())   # keys()
# 之后有其他需求比如要看 key 为 model 的内容有啥
print(content['state_dict'].keys())
