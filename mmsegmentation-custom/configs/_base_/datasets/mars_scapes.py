# dataset settings
dataset_type = 'MarsscapesDataset'
data_root = 'data/mars-scapes'

# 0.x 版本通常需要在这里定义归一化参数
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (256, 512)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    # 0.x 中使用 img_scale, ratio_range 放在 Resize 中
    dict(type='Resize', img_scale=(1024, 512), ratio_range=(0.5, 2.0)),
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
        img_scale=(512, 256),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75], # 如果需要 TTA 可以取消注释
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
    samples_per_gpu=4,  # 对应 1.x 的 batch_size
    workers_per_gpu=2,  # 对应 1.x 的 num_workers
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img/train',       # 对应 1.x 的 img_path
        ann_dir='label_id/train',  # 对应 1.x 的 seg_map_path
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img/val',
        ann_dir='label_id/val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img/test',
        ann_dir='label_id/test',
        pipeline=test_pipeline))

# 0.x 的评估配置
evaluation = dict(interval=2000, metric='mIoU', pre_eval=True)