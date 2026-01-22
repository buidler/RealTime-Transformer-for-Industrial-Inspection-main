"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.

本脚本用于：
1）从 YAML 配置和权重文件构建 RTDETR 推理模型；
2）对单张输入图像进行目标检测（矩形框）；
3）基于检测框做传统图像处理，提取纤维的不规则轮廓并可视化。
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import torch
import torch.nn as nn 
import torchvision.transforms as T

import numpy as np 
from PIL import Image, ImageDraw
from scipy import ndimage

from src.core import YAMLConfig


def _binary_mask_from_crop(crop):
    # 对单个检测框裁剪出的局部图像做灰度+自适应阈值，生成前景二值掩膜
    arr = np.asarray(crop.convert("L"))
    hist, _ = np.histogram(arr, bins=256, range=(0, 255))
    total = arr.size
    sum_total = np.dot(hist, np.arange(256))
    sum_b = 0.0
    w_b = 0.0
    max_var = 0.0
    threshold = 0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = t
    mask = arr >= threshold
    inv_mask = ~mask
    if inv_mask.mean() > mask.mean():
        mask = inv_mask
    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3)))
    mask = ndimage.binary_closing(mask, structure=np.ones((3, 3)))
    return mask


def _trace_contour(boundary):
    h, w = boundary.shape
    ys, xs = np.nonzero(boundary)
    if ys.size == 0:
        return None
    start_idx = np.lexsort((xs, ys))[0]
    y = int(ys[start_idx])
    x = int(xs[start_idx])
    dirs = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
    prev_dir = 4
    contour = []
    first = True
    while True:
        contour.append((x, y))
        found = False
        for k in range(8):
            dir_idx = (prev_dir + 6 + k) % 8
            dy, dx = dirs[dir_idx]
            ny = y + dy
            nx = x + dx
            if 0 <= ny < h and 0 <= nx < w and boundary[ny, nx]:
                y = ny
                x = nx
                prev_dir = dir_idx
                found = True
                break
        if not found:
            break
        if (y == contour[0][1] and x == contour[0][0]) and not first:
            break
        first = False
    if len(contour) < 3:
        return None
    return np.asarray(contour, dtype=np.float32)


def _contours_from_box(im, box):
    # 在整图坐标系下，根据单个检测框提取前景连通域并提取贴合边界的轮廓点集
    x0, y0, x1, y1 = [int(v) for v in box]
    w, h = im.size
    x0 = max(0, min(x0, w - 1))
    x1 = max(0, min(x1, w))
    y0 = max(0, min(y0, h - 1))
    y1 = max(0, min(y1, h))
    if x1 <= x0 or y1 <= y0:
        return []
    crop = im.crop((x0, y0, x1, y1))
    mask = _binary_mask_from_crop(crop)
    labeled, num = ndimage.label(mask)
    contours = []
    for label_id in range(1, num + 1):
        region = labeled == label_id
        if region.sum() < 50:
            continue
        eroded = ndimage.binary_erosion(region, structure=np.ones((3, 3)))
        boundary = region & ~eroded
        contour = _trace_contour(boundary)
        if contour is None or contour.shape[0] < 10:
            continue
        contour[:, 0] += x0
        contour[:, 1] += y0
        contours.append(contour.tolist())
    return contours


def draw(images, labels, boxes, scores, thrh=0.6, draw_contours=False, save_dir=None, im_name=None):
    # 遍历每张图，先根据置信度阈值画红色矩形框，可选叠加黄色不规则轮廓
    # save_dir 指定结果图保存目录；im_name 用于根据原图文件名生成输出文件名
    for i, im in enumerate(images):
        drawer = ImageDraw.Draw(im)

        scr = scores[i]
        keep = scr > thrh
        if keep.sum() == 0:
            continue
        scrs = scr[keep]
        lab = labels[i][keep]
        box = boxes[i][keep]
        order = torch.argsort(scrs, descending=True)
        scrs = scrs[order]
        lab = lab[order]
        box = box[order]

        kept_boxes = []

        for j, b in enumerate(box):
            b_np = b.cpu().numpy()
            if kept_boxes:
                kb = np.stack(kept_boxes, axis=0)
                xx0 = np.maximum(kb[:, 0], b_np[0])
                yy0 = np.maximum(kb[:, 1], b_np[1])
                xx1 = np.minimum(kb[:, 2], b_np[2])
                yy1 = np.minimum(kb[:, 3], b_np[3])
                inter_w = np.clip(xx1 - xx0, a_min=0.0, a_max=None)
                inter_h = np.clip(yy1 - yy0, a_min=0.0, a_max=None)
                inter = inter_w * inter_h
                area_b = (b_np[2] - b_np[0]) * (b_np[3] - b_np[1])
                area_k = (kb[:, 2] - kb[:, 0]) * (kb[:, 3] - kb[:, 1])
                union = area_b + area_k - inter
                iou = inter / np.maximum(union, 1e-6)
                if float(iou.max()) > 0.7:
                    continue
            kept_boxes.append(b_np)

            drawer.rectangle(list(b), outline="red")
            drawer.text((b[0], b[1]), text=f"{lab[j].item()} {round(scrs[j].item(), 2)}", fill="blue")

            if draw_contours:
                contour_list = _contours_from_box(im, b)
                for pts in contour_list:
                    if len(pts) < 3:
                        continue
                    pts_arr = np.asarray(pts, dtype=np.float32)
                    x0, y0, x1, y1 = b_np
                    dist_edge = np.minimum.reduce([
                        pts_arr[:, 0] - x0,
                        pts_arr[:, 1] - y0,
                        x1 - pts_arr[:, 0],
                        y1 - pts_arr[:, 1],
                    ])
                    if (dist_edge < 3.0).mean() > 0.7:
                        continue
                    step = max(1, len(pts_arr) // 400)
                    pts_arr = pts_arr[::step]
                    pts_seq = [(float(p[0]), float(p[1])) for p in pts_arr]
                    drawer.line(pts_seq + [pts_seq[0]], fill="yellow", width=1)

        # 如果未显式指定保存目录，则默认使用当前目录
        if save_dir is None:
            save_dir = "."
        if im_name is not None and len(images) == 1:
            stem, _ = os.path.splitext(os.path.basename(im_name))
            if draw_contours:
                out_name = f"{stem}_contours.jpg"
            else:
                out_name = f"{stem}_det.jpg"
        else:
            out_name = f"results_{i}.jpg"
        out_path = os.path.join(save_dir, out_name)
        im.save(out_path)


def main(args, ):
    """main
    使用 YAML 配置和训练权重构建部署模型，对单张图像进行推理与可视化。

    这里不直接操作 PyTorch 的底层细节，而是通过 YAMLConfig 这个封装类：
    - cfg.model 是根据配置文件构建好的 RTDETR 网络结构（nn.Module 子类）；
    - cfg.postprocessor 包含解码框、阈值过滤、NMS、缩放回原图尺寸等后处理逻辑。
    """
    # 通过 YAMLConfig 读取配置文件，并在其中记录权重路径（resume）
    cfg = YAMLConfig(args.config, resume=args.resume)

    if args.resume:
        # 从磁盘加载 checkpoint（state_dict），map_location='cpu' 表示先加载到 CPU
        checkpoint = torch.load(args.resume, map_location='cpu') 
        # 训练时如果启用了 EMA，这里优先使用 ema 的权重；否则使用普通 model 权重
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        # 当前脚本只支持从权重文件加载，不支持“随机初始化”推理
        raise AttributeError('Only support resume to load model.state_dict by now.')

    # NOTE 加载训练阶段权重，并转换为推理部署模式
    # cfg.model 是一个普通的 PyTorch 模型（nn.Module），这里把参数字典填进去
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            # 部署版 RTDETR 模型和后处理（含 NMS、尺度还原等）
            # deploy() 一般会做一些结构上的简化，例如去掉训练专用分支等
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes):
            # images: 形状为 [1, 3, H, W] 的张量（BCHW），值范围为 0~1
            # orig_target_sizes: 原图宽高，用于把预测框从 640x640 缩放回原图尺寸
            outputs = self.model(images)
            # postprocessor 内部会完成分类得分、边框解码、NMS 等，并返回 numpy 风格结果
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    # 把自定义的 Model 包装成一个普通的 PyTorch 模型，并移动到指定设备（CPU/GPU）
    model = Model().to(args.device)

    # 读入待推理图像，并转换成 RGB（PyTorch 一般假设输入是 3 通道）
    im_pil = Image.open(args.im_file).convert('RGB')
    w, h = im_pil.size
    # orig_size 形状为 [1, 2]，内部是 [width, height]，放到与模型相同的 device 上
    orig_size = torch.tensor([w, h])[None].to(args.device)

    # torchvision.transforms 用于把 PIL 图像转成张量，并缩放到模型期望的大小
    transforms = T.Compose([
        T.Resize((640, 640)),  # 统一缩放到 640x640
        T.ToTensor(),          # 转成 [C, H, W]，像素值从 0~255 归一化到 0~1 的 float32
    ])
    # im_data 形状为 [1, 3, 640, 640]，在最前面增加 batch 维度
    im_data = transforms(im_pil)[None].to(args.device)

    # 前向推理，返回的是 (labels, boxes, scores)
    output = model(im_data, orig_size)
    labels, boxes, scores = output

    # 确定结果图保存路径：
    # 1）优先使用命令行传入的 --save-dir；
    # 2）否则在权重文件同级目录下创建 contour_results 子目录；
    # 3）如果没有权重路径（理论上不会发生），则退回到当前工作目录下的 contour_results。
    if args.save_dir:
        save_dir = args.save_dir
    else:
        if args.resume:
            ckpt_dir = os.path.dirname(args.resume)
            save_dir = os.path.join(ckpt_dir, "contour_results")
        else:
            save_dir = "contour_results"
    os.makedirs(save_dir, exist_ok=True)

    # 把 PIL 原图和推理结果送入可视化函数，生成带矩形框/轮廓的结果图
    draw([im_pil], labels, boxes, scores, thrh=args.score_thr, draw_contours=args.contours, save_dir=save_dir, im_name=args.im_file)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # -c: 配置文件路径（例如 configs/rtdetrv2/rtdetrv2_fiber.yml）
    parser.add_argument('-c', '--config', type=str)
    # -r: 训练好的模型权重路径（.pth）
    parser.add_argument('-r', '--resume', type=str)
    # -f: 单张输入图像路径
    parser.add_argument('-f', '--im-file', type=str)
    # -d: 使用的设备，cpu 或 cuda:0 等
    parser.add_argument('-d', '--device', type=str, default='cpu')
    # --score-thr: 置信度阈值，控制画哪些检测框
    parser.add_argument('--score-thr', type=float, default=0.6)
    # --contours: 是否启用基于传统图像处理的不规则轮廓绘制
    parser.add_argument('--contours', action='store_true')
    # --save-dir: 结果图保存目录，默认自动放到权重同级目录的 contour_results 中
    parser.add_argument('--save-dir', type=str, default=None)
    args = parser.parse_args()
    main(args)
