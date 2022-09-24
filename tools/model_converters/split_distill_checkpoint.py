import argparse
import os

import torch
import mmcv
import sys
from mmrazor.models.builder import build_algorithm
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcls.models import build_classifier
from mmdet.models import build_detector


def split_student_model(cfg_path, model_path, device='cuda', save_path=None):
    """
    :param: cfg_path: your normal config file path which is not disitilation cfg path
    :param: model_path: your distilation checkpoint path
    :param: save_path: student model save path
    """
    cfg = mmcv.Config.fromfile(cfg_path)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True
    
    model = build_detector(cfg.model)
    model_ckpt = torch.load(model_path)
    
    pretrained_dict = model_ckpt['state_dict']
    model_dict = model.state_dict()
    new_dict = {k.replace('architecture.model.', ''): v for k, v in pretrained_dict.items() if k.replace('architecture.model.', '') in model_dict.keys()}
    model_dict.update(new_dict)
    model.load_state_dict(model_dict)
    torch.save({'state_dict': model.state_dict(), 'meta': model_ckpt['meta'],
                'optimizer': model_ckpt['optimizer']}, save_path)
    
def parse_args():
    parser = argparse.ArgumentParser(description='Split student checkpoint from distiller checkpoint')
    parser.add_argument('config', help='model config file path,not distiller config path')
    parser.add_argument('model_path', help='distiller checkpoint path')
    parser.add_argument('--output_path', help='the checkpoint file to resume from')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    output_path = args.output_path
    if args.output_path == None:
        output_path = args.model_path

    split_student_model(args.config,args.model_path,save_path=output_path)

