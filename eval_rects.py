#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_rects.py

负责根据检测到的矩形框和 ground truth 计算：
- 正确个数 correct
- 错误个数 wrong（检测出来但中心不在任何 GT 框中）
- 检测总数 total_detected
- ground truth 中的 Rect 总数 total_gt
- 正确率 accuracy = correct / total_gt

可被训练脚本和测试脚本共同调用。
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple
import re

@dataclass
class Rect:
    x1: int
    y1: int
    x2: int
    y2: int
    matched: bool = False  # 用于标记是否已经被匹配


# 匹配 Rect: (x1, y1, x2, y2) —— 行尾可以有其它内容（Point 等）
RECT_PATTERN = re.compile(
    r"Rect:\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)"
)

# 匹配 Image: 后面的整行路径
IMAGE_PATTERN = re.compile(r"Image:\s*(.+)")


def rect_contains_point(rect: Rect, x: float, y: float) -> bool:
    """判断点 (x, y) 是否在矩形 rect 内（含边界）。"""
    return rect.x1 <= x <= rect.x2 and rect.y1 <= y <= rect.y2


def parse_gt_file(gt_file: str) -> Dict[str, List[Rect]]:
    """
    解析 ground truth 文件，只提取 Rect 信息。

    文件格式示例：
        Image: data/2023-12-06-11-10-47.jpg
        Rect: (204, 405, 284, 510)  Point: (245, 461)
        Rect: (446, 404, 524, 499)  Point: (488, 453)

    返回:
        {normalized_image_path: [Rect, Rect, ...], ...}
        注意：键已经过 os.path.normpath 处理
    """
    gt_rects: Dict[str, List[Rect]] = {}
    current_image = None

    # utf-8-sig 可以自动去掉 UTF-8 BOM
    with open(gt_file, "r", encoding="utf-8-sig") as f:
        for line in f:
            # 同时去掉常见的 BOM 残留（保险起见）
            line = line.replace("\ufeff", "").strip()
            if not line:
                continue

            img_match = IMAGE_PATTERN.search(line)
            if img_match:
                raw_path = img_match.group(1).strip()
                # 路径正规化，解决 / 与 \、相对路径等问题
                current_image = os.path.normpath(raw_path)
                if current_image not in gt_rects:
                    gt_rects[current_image] = []
                continue

            rect_match = RECT_PATTERN.search(line)
            if rect_match and current_image:
                x1, y1, x2, y2 = map(int, rect_match.groups())
                gt_rects[current_image].append(Rect(x1, y1, x2, y2))

    return gt_rects


def evaluate_boxes_for_image(
    boxes: List[Tuple[int, int, int, int]],
    image_path: str,
    gt_rects: Dict[str, List[Rect]],
):
    """
    对“单张图”的检测结果进行评估。

    参数:
        boxes: [(x1, y1, x2, y2), ...]  检测出来的矩形框
        image_path: 该图片的路径字符串；本函数内部会做 normpath
        gt_rects: parse_gt_file() 的返回结果（键也已做过 normpath）

    返回:
        correct, wrong, total_detected, total_gt, accuracy
    """
    # 与 GT 里的 Image: 路径对齐
    image_key = os.path.normpath(image_path)

    raw_list = gt_rects.get(image_key, [])
    # 拷贝一份，避免 matched 状态被污染
    gt_list = [Rect(r.x1, r.y1, r.x2, r.y2, False) for r in raw_list]

    total_gt = len(gt_list)
    total_detected = len(boxes)
    correct = 0
    wrong = 0

    # 没有 GT 框，则所有检测都算错误
    if total_gt == 0:
        wrong = total_detected
        accuracy = 0.0
        return correct, wrong, total_detected, total_gt, accuracy

    for (x1, y1, x2, y2) in boxes:
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        matched_any = False

        for g_rect in gt_list:
            if (not g_rect.matched) and rect_contains_point(g_rect, cx, cy):
                g_rect.matched = True
                correct += 1
                matched_any = True
                break

        if not matched_any:
            wrong += 1

    accuracy = correct / total_gt if total_gt > 0 else 0.0
    return correct, wrong, total_detected, total_gt, accuracy
