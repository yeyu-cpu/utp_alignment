#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
多张图测试入口脚本（参数来自 JSON，带 tqdm 进度条）。

列表 txt 格式：
    每行：目标图路径 模板图路径 gt文件路径

说明：
    - 测试参数从 --param_json 指定的 JSON 文件导入；
    - 启动时在日志中输出：参数 JSON 路径 + 具体参数值；
    - 测试过程中只显示一个 tqdm 进度条，不输出每张图的测试信息；
    - 每张图的可视化结果 / 框 txt / 评估 txt 都保存在 out_dir 中；
    - ✅ 测试结束后：
        * 在控制台打印整体平均正确率（micro-average）
        * 在 out_dir 下保存 test_summary.txt
"""

import os
import argparse
import json

from tqdm import trange

import template_matching_core as core


def read_list_file(list_path: str):
    items = []
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            target_path, template_path, gt_path = parts[:3]
            items.append((target_path, template_path, gt_path))
    return items


def load_params_from_json(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    th = float(data["MATCHING_THRESHOLD"])
    wc = float(data["WEIGHT_COLOR"])
    wt = float(data["WEIGHT_TEXTURE"])
    wm = float(data["WEIGHT_MSE"])
    ws = float(data["WEIGHT_SSIM"])

    return th, wc, wt, wm, ws


def apply_params_to_core(th, wc, wt, wm, ws):
    core.MATCHING_THRESHOLD = th
    core.WEIGHT_COLOR = wc
    core.WEIGHT_TEXTURE = wt
    core.WEIGHT_MSE = wm
    core.WEIGHT_SSIM = ws


def main():
    parser = argparse.ArgumentParser(description="多张图测试（参数来自 JSON）")
    parser.add_argument(
        "--list", default="dataset/test_list.txt",
        help="测试列表 txt，每行：目标图路径 模板图路径 gt文件路径"
    )
    parser.add_argument(
        "--out_dir", default="results/test_results",
        help="保存测试结果的目录"
    )
    parser.add_argument(
        "--param_json", default="results/best_params.json",
        help="测试时使用的参数 JSON 文件路径"
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    items = read_list_file(args.list)
    if not items:
        print(f"[WARN] 列表文件中没有有效数据行: {args.list}")
        return

    # 1. 读取参数
    th, wc, wt, wm, ws = load_params_from_json(args.param_json)

    # 2. 打印参数日志
    print("========= Test 参数设置 ==========")
    print(f"参数 JSON 路径: {args.param_json}")
    print(f"MATCHING_THRESHOLD = {th:.4f}")
    print(f"WEIGHT_COLOR       = {wc:.4f}")
    print(f"WEIGHT_TEXTURE     = {wt:.4f}")
    print(f"WEIGHT_MSE         = {wm:.4f}")
    print(f"WEIGHT_SSIM        = {ws:.4f}")
    print("================================\n")

    apply_params_to_core(th, wc, wt, wm, ws)

    # ✅ micro-average 统计
    sum_correct = 0
    sum_gt = 0
    valid_cnt = 0

    for idx in trange(len(items), desc="Testing", ncols=80):
        target, template, gt = items[idx]
        prefix = f"test_{idx:04d}"

        core.set_paths(
            template_path=template,
            target_path=target,
            gt_path=gt,
            output_dir=args.out_dir,
            prefix=prefix,
        )

        eval_txt_path = os.path.join(args.out_dir, f"{prefix}_eval.txt")

        correct, wrong, total_detected, total_gt, acc = core.test(
            eval_save_path=eval_txt_path,
            log=False
        )

        if total_gt is not None and total_gt > 0:
            sum_correct += int(correct)
            sum_gt += int(total_gt)
            valid_cnt += 1

    # ===== 汇总结果 =====
    summary_path = os.path.join(args.out_dir, "test_summary.txt")

    if sum_gt > 0:
        avg_acc = sum_correct / sum_gt

        # 控制台输出
        print("\n========= Test 汇总 =========")
        print(f"有效统计图片数（GT>0）: {valid_cnt}/{len(items)}")
        print(f"总 True Positive(correct) = {sum_correct}")
        print(f"总 GT(total_gt)          = {sum_gt}")
        print(f"平均正确率(sum_correct/sum_gt) = {avg_acc:.4f}")
        print("============================\n")

        # ✅ 保存到 txt
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"Test list: {args.list}\n")
            f.write(f"Param json: {args.param_json}\n\n")
            f.write(f"Valid images (GT > 0): {valid_cnt} / {len(items)}\n")
            f.write(f"Total correct (TP): {sum_correct}\n")
            f.write(f"Total GT: {sum_gt}\n")
            f.write(f"Average accuracy (sum_correct / sum_gt): {avg_acc:.6f}\n")

    else:
        print("\n[WARN] 所有图片的 GT 总数为 0，无法计算平均正确率。\n")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("No valid GT found. Average accuracy cannot be computed.\n")


if __name__ == "__main__":
    main()
