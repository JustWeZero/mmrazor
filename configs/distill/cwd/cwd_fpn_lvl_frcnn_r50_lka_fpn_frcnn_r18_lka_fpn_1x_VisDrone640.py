_base_ = [
    '../../_base_/datasets/mmdet/VisDrone_detection_640x640_overlap40.py',
    '../../_base_/schedules/mmdet/schedule_1x.py',
    '../../_base_/mmdet_runtime.py'
]

train_cfg = dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_across_levels=False,
            nms_pre=2000,
            nms_post=1000,
            max_num=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False))
test_cfg=dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=1000,
        nms_post=1000,
        max_per_img=-1,
        nms=dict(type='nms', iou_threshold=0.7),
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05,
        nms=dict(type='soft_nms', iou_threshold=0.3, min_score=0.05),
        max_per_img=-1,
        do_tile_as_aug=False
    )  #
    # soft-nms is also supported for rcnn testing
    # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
)
# test_cfg=dict(
#         nms_pre=1000,
#         min_bbox_size=0,
#         score_thr=0.05,
#         nms=dict(type='soft_nms', iou_threshold=0.3, min_score=0.05),
#         max_per_img=-1)

# model settings
student = dict(
    type='mmdet.FasterRCNN',
    backbone=dict(
        type='ResNet',
        depth=18,
        norm_eval=True,
        norm_cfg=dict(type='BN'),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18'),
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'
    ),
    neck=dict(
        type='lka_FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=64,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=64,
        feat_channels=64,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[2],  # [8]
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=64,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=64,
            fc_out_channels=256,
            roi_feat_size=7,
            num_classes=10,  # 80
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    train_cfg = train_cfg,
    test_cfg = test_cfg
)
    # # training and testing settings
    # train_cfg=dict(
    #     assigner=dict(type='ATSSAssigner', topk=9),
    #     allowed_border=-1,
    #     pos_weight=-1,
    #     debug=False),
    # test_cfg=dict(
    #     nms_pre=1000,
    #     min_bbox_size=0,
    #     score_thr=0.05,
    #     nms=dict(type='nms', iou_threshold=0.6),
    #     max_per_img=100)

# checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_r101_fpn_mstrain_2x_coco/gfl_r101_fpn_mstrain_2x_coco_20200629_200126-dd12f847.pth'  # noqa: E501
checkpoint = r'/home/group5/lzj/VisDrone_cache/lka_fpn/faster_rcnn_r50_lka_fpn_1x_VisDrone640/slice_640x640_lr0.02_1x_1g/epoch_12.pth'

teacher = dict(
    type='mmdet.FasterRCNN',
    # pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN'),  # requires_grad=True
        norm_eval=True,
        style='pytorch'
        ),
    neck=dict(
        type='lka_FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[2],  # [8]
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=10,  # 80
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
        train_cfg=train_cfg,
        test_cfg=test_cfg)

# algorithm setting
algorithm = dict(
    type='GeneralDistill',
    architecture=dict(
        type='MMDetArchitecture',
        model=student,
    ),
    distiller=dict(
        type='SingleTeacherDistiller',
        teacher=teacher,
        teacher_trainable=False,
        components=[
            dict(
                student_module='neck.fpn_convs.3.conv',
                teacher_module='neck.fpn_convs.3.conv',
                losses=[
                    dict(
                        type='ChannelWiseDivergence',
                        name='loss_cwd_fpn_3',
                        tau=1,
                        loss_weight=5,
                    )
                ]),
            dict(
                student_module='neck.fpn_convs.2.conv',
                teacher_module='neck.fpn_convs.2.conv',
                losses=[
                    dict(
                        type='ChannelWiseDivergence',
                        name='loss_cwd_fpn_2',
                        tau=1,
                        loss_weight=5,
                    )
                ]),
            dict(
                student_module='neck.fpn_convs.1.conv',
                teacher_module='neck.fpn_convs.1.conv',
                losses=[
                    dict(
                        type='ChannelWiseDivergence',
                        name='loss_cwd_fpn_1',
                        tau=1,
                        loss_weight=5,
                    )
                ]),
            dict(
                student_module='neck.fpn_convs.0.conv',
                teacher_module='neck.fpn_convs.0.conv',
                losses=[
                    dict(
                        type='ChannelWiseDivergence',
                        name='loss_cwd_fpn_0',
                        tau=1,
                        loss_weight=5,
                    )
                ])
        ]),
)
evaluation = dict(interval=1, metric='bbox',save_best='bbox_mAP')

find_unused_parameters = True

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
