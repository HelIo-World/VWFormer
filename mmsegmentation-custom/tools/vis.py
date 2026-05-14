import os
import cv2
import numpy as np
import argparse
import torch
# 注意：0.x 版本的导入路径与 1.x 不同
from mmseg.apis import init_segmentor, inference_segmentor

# 你的火星数据集类别和调色板
CLASSES = ('soil', 'bedrock', 'gravel', 'sand', 
           'big rock', 'sky', 'ridge', 'rover', 'unknown')
PALETTE = [[0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 0, 255],
           [255, 255, 0], [34, 56, 19], [128, 128, 128], [0, 85, 0], [170, 85, 0]]

def parse_args():
    parser = argparse.ArgumentParser(description='mmsegmentation 0.x 批量生成彩色预测图')
    parser.add_argument('--config', type=str, required=True, help='模型配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型权重文件路径')
    parser.add_argument('--model-name', type=str, required=True, help='模型名称（如 dpanet，将作为输出文件夹名）')
    parser.add_argument('--img-dir', type=str, default='data/mars-scapes/img/test', help='原始图片存放目录')
    parser.add_argument('--out-dir', type=str, default='vis', help='预测结果保存的基础目录（默认：vis）')
    parser.add_argument('--device', type=str, default='cuda:0', help='推理使用的设备（默认：cuda:0）')
    return parser.parse_args()

def colorize_mask(mask, palette):
    """
    将单通道类别索引图转为彩色图
    mask: numpy array (H, W)
    """
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_mask[mask == label] = color
    # OpenCV 使用 BGR 顺序保存，因此这里由 RGB 转 BGR
    return cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)

def main():
    args = parse_args()
    
    # 准备输出路径
    save_path = os.path.join(args.out_dir, args.model_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 1. 初始化模型 (mmseg 0.x API)
    device = args.device if torch.cuda.is_available() else 'cpu'
    model = init_segmentor(args.config, args.checkpoint, device=device)

    # 2. 获取图片列表
    img_list = [f for f in os.listdir(args.img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"找到 {len(img_list)} 张图片，开始在 {args.device} 上推理模型: {args.model_name}")

    for img_name in img_list:
        full_img_path = os.path.join(args.img_dir, img_name)
        
        # 3. 执行推理 (mmseg 0.x 返回的是 list[numpy.ndarray])
        # 对于单张图推理，result 是长度为 1 的 list，内容是 (H, W) 的掩码
        result = inference_segmentor(model, full_img_path)
        pred_mask = result[0] 
        
        # 4. 上色
        color_img = colorize_mask(pred_mask, PALETTE)
        
        # 5. 保存结果 (保持与原图同名或转为png)
        out_name = os.path.splitext(img_name)[0] + '.png'
        cv2.imwrite(os.path.join(save_path, out_name), color_img)

    print(f"所有结果已保存至: {save_path}")

if __name__ == '__main__':
    main()