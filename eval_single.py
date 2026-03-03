#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
根据测试结果文件与 ground truth 文件计算检测正确率。

你只需要修改下面两个变量：
    TEST_FILE = "/absolute/path/to/test_result.txt"
    GT_FILE   = "/absolute/path/to/ground_truth.txt"
然后直接运行本文件即可。
"""

import re
from dataclasses import dataclass
from typing import Dict, List


# ==========================
# 在此填写两个文件的完整路径
# ==========================
TEST_FILE = r"outputs_template_matching/result_gc_2023-12-06-11-10-47.jpg.txt"      # ← 测试结果文件
GT_FILE   = r"annotations/annotations_2023-12-06-11-10-47.jpg.txt"     # ← ground truth 文件
# ==========================


@dataclass
class Rect:
    x1: int
    y1: int
    x2: int
    y2: int
    matched: bool = False  # ground truth 里用于标记是否已经被匹配


RECT_PATTERN = re.compile(
    r"Rect:\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)"
)
IMAGE_PATTERN = re.compile(r"Image:\s*(.+)")


def parse_rect_files(test_file: str, gt_file: str):
    """解析测试文件与 GT 文件，按 Image 分组返回 Rect 列表。"""
    test_rects: Dict[str, List[Rect]] = {}
    gt_rects: Dict[str, List[Rect]] = {}

    # 解析 ground truth
    current_image = None
    with open(gt_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            img_match = IMAGE_PATTERN.search(line)
            if img_match:
                current_image = img_match.group(1).strip()
                if current_image not in gt_rects:
                    gt_rects[current_image] = []
                continue

            rect_match = RECT_PATTERN.search(line)
            if rect_match and current_image:
                x1, y1, x2, y2 = map(int, rect_match.groups())
                gt_rects[current_image].append(Rect(x1, y1, x2, y2))

    # 解析测试结果
    current_image = None
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            img_match = IMAGE_PATTERN.search(line)
            if img_match:
                current_image = img_match.group(1).strip()
                if current_image not in test_rects:
                    test_rects[current_image] = []
                continue

            rect_match = RECT_PATTERN.search(line)
            if rect_match and current_image:
                x1, y1, x2, y2 = map(int, rect_match.groups())
                test_rects[current_image].append(Rect(x1, y1, x2, y2))

    return test_rects, gt_rects


def rect_contains_point(rect: Rect, x: float, y: float) -> bool:
    """判断点 (x, y) 是否在矩形 rect 内。"""
    return rect.x1 <= x <= rect.x2 and rect.y1 <= y <= rect.y2


def evaluate(test_rects: Dict[str, List[Rect]],
             gt_rects: Dict[str, List[Rect]]):
    """
    计算：
    - 正确个数 correct
    - 错误个数 wrong
    - 测试文件检测出的总数 total_detected
    - ground truth 中的矩形总数 total_gt
    - 正确率 accuracy
    """
    total_gt = sum(len(rects) for rects in gt_rects.values())
    total_detected = sum(len(rects) for rects in test_rects.values())

    correct = 0
    wrong = 0

    for image, t_rects in test_rects.items():
        # 该图片没有 GT，所有检测都算错误
        if image not in gt_rects or len(gt_rects[image]) == 0:
            wrong += len(t_rects)
            continue

        gt_list = gt_rects[image]

        for t_rect in t_rects:
            cx = (t_rect.x1 + t_rect.x2) / 2.0
            cy = (t_rect.y1 + t_rect.y2) / 2.0

            matched_any = False
            for g_rect in gt_list:
                if (not g_rect.matched) and rect_contains_point(g_rect, cx, cy):
                    g_rect.matched = True
                    correct += 1
                    matched_any = True
                    break

            if not matched_any:
                wrong += 1

    accuracy = correct / total_gt if total_gt else 0.0
    return correct, wrong, total_detected, total_gt, accuracy


def main():
    print("读取文件：")
    print("  测试结果文件:", TEST_FILE)
    print("  Ground Truth 文件:", GT_FILE)

    test_rects, gt_rects = parse_rect_files(TEST_FILE, GT_FILE)
    correct, wrong, total_detected, total_gt, accuracy = evaluate(test_rects, gt_rects)

    print("\n===== 结果 =====")
    print(f"Ground truth中矩形框总数 (GT count): {total_gt}")
    print(f"检测出的矩形框总数 (Detected count): {total_detected}")
    print(f"检测正确的个数 (True Positive): {correct}")
    print(f"检测错误的个数 (False Positive): {wrong}")
    print(f"正确率 (correct / total_gt): {accuracy:.4f}")


if __name__ == "__main__":
    main()