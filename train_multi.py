#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
多张图训练入口脚本。

列表 txt 格式：
    每行：目标图路径 模板图路径 gt文件路径
        例如：
        data/img1.jpg data_templates/tmpl1.png annotations/img1.txt

运行示例：
    python train_multi.py \
        --list train_list.txt \
        --out_dir outputs_train_multi \
        --json_out outputs_train_multi/train_params.json \
        --num_iters 300

说明：
    - 对列表中的每张图独立做一次 train（SPSA+Adam）；
    - 每张图结束后保存：可视化图 / 框 txt / metrics txt；
    - 所有图训练结束后：
        * 计算所有图“最佳 accuracy”的平均值；
        * 计算所有图“最佳参数”的平均值（五个参数），并写到 json 顶层：
              MATCHING_THRESHOLD
              WEIGHT_COLOR
              WEIGHT_TEXTURE
              WEIGHT_MSE
              WEIGHT_SSIM
        * json 中还会包含 num_images / mean_best_value / per_image 等信息。
"""

import os
import json
import argparse

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


def main():
    parser = argparse.ArgumentParser(description="多张图训练（SPSA+Adam 调参）")
    parser.add_argument(
        "--list", required=False,
        default="dataset/train_list.txt",
        help="训练列表 txt，每行：目标图路径 模板图路径 gt文件路径"
    )
    parser.add_argument(
        "--out_dir", required=False,
        default="results/train_results",
        help="保存每张图可视化结果和框 txt 的目录"
    )
    parser.add_argument(
        "--json_out", required=False,
        default="saresults/best_params.json",
        help="保存平均最优值（accuracy）的 json 文件路径"
    )
    parser.add_argument(
        "--num_iters", type=int, default=300,
        help="每张图的训练迭代次数（默认 300）"
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    items = read_list_file(args.list)
    if not items:
        print(f"[WARN] 列表文件中没有有效数据行: {args.list}")
        return

    per_image = []

    # 用于累积“每张图最佳参数”，后面求平均
    sum_th = 0.0
    sum_wc = 0.0
    sum_wt = 0.0
    sum_wm = 0.0
    sum_ws = 0.0

    total = len(items)  # ⭐ 新增：总图像数量，方便打印进度

    # ⭐ 改动：enumerate 从 1 开始，这样 idx 就是“第几张图”
    for idx, (target, template, gt) in enumerate(items, start=1):
        # ⭐ 新增：每轮训练开始前打印当前进度
        print(f"[INFO] 正在处理第 {idx}/{total} 张图像: {target}")

        prefix = f"train_{idx:04d}"

        # 设置当前这一张图的路径和输出前缀
        core.set_paths(
            template_path=template,
            target_path=target,
            gt_path=gt,
            output_dir=args.out_dir,
            prefix=prefix,
        )

        # 调参：行为同原来的 train，只是多返回了 best 信息
        result = core.train(num_iters=args.num_iters, return_best=True)

        best_loss = result["best_loss"]
        c, w, td, tg, acc = result["best_metrics"]
        best_theta = result["best_theta"]

        # 从 best_theta 还原出这张图的最佳五个参数
        # 注意：_theta_to_params 在 core 里已经定义，这里直接用它即可，不改变核心逻辑
        th_i, wc_i, wt_i, wm_i, ws_i = core._theta_to_params(best_theta)

        # 累加，后面求平均
        sum_th += th_i
        sum_wc += wc_i
        sum_wt += wt_i
        sum_wm += wm_i
        sum_ws += ws_i

        # 调参结束后，用最佳参数再跑一次检测，保存可视化图和框 txt
        _ = core.run_detection(
            template_path=template,
            target_path=target,
            save_visual=True,
            save_txt=True,
            log=False,   # 训练阶段不再额外打印
        )

        # 每张图的最优结果写到一个单独的 metrics txt 文件
        metrics_txt_path = os.path.join(
            args.out_dir,
            f"{prefix}_metrics.txt"
        )
        with open(metrics_txt_path, "w", encoding="utf-8") as f:
            f.write(f"image: {target}\n")
            f.write(f"template: {template}\n")
            f.write(f"gt: {gt}\n")
            f.write(f"best_loss: {best_loss:.4f}\n")
            f.write(f"correct: {c}\n")
            f.write(f"wrong: {w}\n")
            f.write(f"detected: {td}\n")
            f.write(f"gt_total: {tg}\n")
            f.write(f"accuracy: {acc:.4f}\n")
            f.write(f"MATCHING_THRESHOLD: {th_i:.4f}\n")
            f.write(f"WEIGHT_COLOR: {wc_i:.4f}\n")
            f.write(f"WEIGHT_TEXTURE: {wt_i:.4f}\n")
            f.write(f"WEIGHT_MSE: {wm_i:.4f}\n")
            f.write(f"WEIGHT_SSIM: {ws_i:.4f}\n")

        per_image.append({
            "image": target,
            "template": template,
            "gt": gt,
            "best_loss": best_loss,
            "correct": c,
            "wrong": w,
            "detected": td,
            "gt_total": tg,
            "accuracy": acc,
            "MATCHING_THRESHOLD": th_i,
            "WEIGHT_COLOR": wc_i,
            "WEIGHT_TEXTURE": wt_i,
            "WEIGHT_MSE": wm_i,
            "WEIGHT_SSIM": ws_i,
            "metrics_txt": metrics_txt_path,
            "visual_path": os.path.join(args.out_dir, f"{prefix}_{os.path.basename(target)}.png"),
            "rects_txt": os.path.join(args.out_dir, f"{prefix}_{os.path.basename(target)}.txt"),
        })

    # 计算平均最优值（这里采用 accuracy 的平均）
    if per_image:
        mean_acc = sum(d["accuracy"] for d in per_image) / len(per_image)

        # 计算五个参数的平均值（权重再归一化）
        n = float(len(per_image))
        mean_th = sum_th / n
        mean_wc = sum_wc / n
        mean_wt = sum_wt / n
        mean_wm = sum_wm / n
        mean_ws = sum_ws / n

        # 确保四个权重归一化到和为 1
        w_sum = mean_wc + mean_wt + mean_wm + mean_ws
        if w_sum > 1e-8:
            mean_wc /= w_sum
            mean_wt /= w_sum
            mean_wm /= w_sum
            mean_ws /= w_sum
    else:
        mean_acc = 0.0
        # 给一个默认值，避免除零
        mean_th = 0.5
        mean_wc = 0.25
        mean_wt = 0.25
        mean_wm = 0.25
        mean_ws = 0.25

    # 组织 json 内容
    # 顶层直接包含 5 个最佳参数，方便 test_multi 直接读取使用
    summary = {
        "MATCHING_THRESHOLD": float(mean_th),
        "WEIGHT_COLOR": float(mean_wc),
        "WEIGHT_TEXTURE": float(mean_wt),
        "WEIGHT_MSE": float(mean_wm),
        "WEIGHT_SSIM": float(mean_ws),
        # "num_images": len(per_image),
        # "metric": "accuracy",
        # "mean_best_value": float(mean_acc),
        # "per_image": per_image,
    }

    os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 按你的要求：控制台只输出 core.train 内部的内容，这里不再额外 print


if __name__ == "__main__":
    main()
