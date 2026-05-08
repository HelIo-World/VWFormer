# dataset settings
dataset_type = 'MarsSegDataset'
data_root = 'data/mars_seg'

# 0.x 必须显式定义归一化参数，通常模型默认使用这些均值和方差
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 576)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    # 1.x 的 RandomResize 对应 0.x Resize + img_scale + ratio_range
    dict(type='Resize', img_scale=(1152, 1024), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=crop_size, # 建议设置为训练时的原图大小或 1.x 里的 scale
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=8,  # 对应 1.x train_dataloader 的 batch_size
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img/train',
        ann_dir='label_id/train',
        img_suffix='.png',
        seg_map_suffix='_labelTrainIds.png',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img/val',
        ann_dir='label_id/val',
        img_suffix='.png',
        seg_map_suffix='_labelTrainIds.png',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img/test',
        ann_dir='label_id/test',
        img_suffix='.png',
        seg_map_suffix='_labelTrainIds.png',
        pipeline=test_pipeline))