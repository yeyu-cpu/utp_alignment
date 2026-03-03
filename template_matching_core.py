#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
单张图的模板匹配 + 超参数优化（SPSA + Adam）。

功能：
- run_detection(): 用当前参数跑一次检测，返回所有框坐标
- train(num_iters=...): 使用 SPSA + Adam 优化 MATCHING_THRESHOLD & 四个权重
    * 训练过程中只有 tqdm 进度条，不打印其它信息
    * 训练结束后只打印“历史最佳那一轮”的结果和参数
- test(): 用当前参数跑一次检测 + 评估，并输出详细结果（可选写入 txt）

本文件作为核心模块使用：
    import template_matching_core as core

多图训练 / 测试请使用单独的入口脚本：
    train_multi.py / test_multi.py
"""

import os
import math
from typing import List, Tuple, Optional

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import trange

from eval_rects import parse_gt_file, evaluate_boxes_for_image

# ===============================
# 配置区域（默认值，可被 set_paths 覆盖）
# ===============================
TEMPLATE_PATH = "dataset/data_templates/crop_20251202_151623.png"    # 模板图
TARGET_PATH   = "dataset/data/2023-12-06-11-10-47.jpg"               # 目标大图

OUTPUT_PATH     = f"outputs_template_matching/result_gc_{os.path.basename(TARGET_PATH)}.png"
OUTPUT_TXT_PATH = f"outputs_template_matching/result_gc_{os.path.basename(TARGET_PATH)}.txt"

# ground truth txt 路径
GT_PATH = "dataset/annotations/annotations_2023-12-06-11-10-47.jpg.txt"

# 初始超参数（会作为训练起点）
MATCHING_THRESHOLD = 0.5244
WEIGHT_COLOR   = 0.2401
WEIGHT_TEXTURE = 0.3028
WEIGHT_MSE     = 0.2429
WEIGHT_SSIM    = 0.2142

# NMS 最大允许重叠（IoU）
MAX_OVERLAP_IOU = 0.4

# Sobel 纹理提取参数
SOBEL_KSIZE = 3
# ===============================


def set_paths(
    template_path: str,
    target_path: str,
    gt_path: str,
    output_dir: str,
    prefix: str = "result",
):
    """
    统一设置当前这一组数据的路径和输出目录。

    这样多图训练 / 测试时，只需要在每一行数据前调用一次：
        set_paths(template, target, gt, out_dir, prefix)

    不会改变原有单图调用方式，只是新增一个方便的接口。
    """
    global TEMPLATE_PATH, TARGET_PATH, GT_PATH, OUTPUT_PATH, OUTPUT_TXT_PATH

    TEMPLATE_PATH = template_path
    TARGET_PATH = target_path
    GT_PATH = gt_path

    os.makedirs(output_dir, exist_ok=True)
    base = os.path.basename(TARGET_PATH)

    OUTPUT_PATH = os.path.join(output_dir, f"{prefix}_{base}.png")
    OUTPUT_TXT_PATH = os.path.join(output_dir, f"{prefix}_{base}.txt")


def non_max_suppression(boxes, scores, iou_threshold=0.5):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def compute_texture(gray):
    gray = gray.astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=SOBEL_KSIZE)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=SOBEL_KSIZE)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return mag


def compute_mse_norm(patch, template):
    patch_f = patch.astype(np.float32)
    tmpl_f  = template.astype(np.float32)
    diff = patch_f - tmpl_f
    mse = np.mean(diff ** 2)
    mse_norm = mse / (255.0 ** 2 + 1e-6)
    mse_norm = float(np.clip(mse_norm, 0.0, 1.0))
    return mse_norm


def compute_ssim_val(patch, template):
    val = ssim(template, patch, data_range=255)
    return float(val)


def normalize_01(x):
    x_min, x_max = x.min(), x.max()
    if x_max - x_min < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return (x - x_min) / (x_max - x_min)


# ===============================
# 核心检测函数（单张图）
# ===============================
def run_detection(
    template_path: str = TEMPLATE_PATH,
    target_path: str = TARGET_PATH,
    save_visual: bool = True,
    save_txt: bool = True,
    log: bool = True,
) -> List[Tuple[int, int, int, int]]:
    """
    用当前全局参数跑一次模板匹配检测，并返回所有矩形框。
    log=False 时不打印任何信息（用于训练阶段）。
    """
    global MATCHING_THRESHOLD, WEIGHT_COLOR, WEIGHT_TEXTURE, WEIGHT_MSE, WEIGHT_SSIM

    weight_sum = WEIGHT_COLOR + WEIGHT_TEXTURE + WEIGHT_MSE + WEIGHT_SSIM
    if abs(weight_sum - 1.0) > 1e-6 and log:
        print(f"[警告] 四个权重之和为 {weight_sum:.6f}，建议调整为 1.0。")

    template = cv2.imread(template_path, cv2.IMREAD_COLOR)
    target   = cv2.imread(target_path,   cv2.IMREAD_COLOR)

    if template is None:
        raise FileNotFoundError(f"无法读取模板图像: {template_path}")
    if target is None:
        raise FileNotFoundError(f"无法读取目标图像: {target_path}")

    th, tw = template.shape[:2]
    H,  W  = target.shape[:2]
    if log:
        print(f"[INFO] 模板尺寸: {tw}x{th}, 目标图尺寸: {W}x{H}")
        print(
            f"[INFO] 当前参数: TH={MATCHING_THRESHOLD:.3f}, "
            f"W_COLOR={WEIGHT_COLOR:.3f}, W_TEXTURE={WEIGHT_TEXTURE:.3f}, "
            f"W_MSE={WEIGHT_MSE:.3f}, W_SSIM={WEIGHT_SSIM:.3f}"
        )

    template_color = template.astype(np.float32)
    target_color   = target.astype(np.float32)

    color_res = cv2.matchTemplate(target_color, template_color, cv2.TM_CCOEFF_NORMED)

    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    target_gray   = cv2.cvtColor(target,   cv2.COLOR_BGR2GRAY)

    template_tex = compute_texture(template_gray)
    target_tex   = compute_texture(target_gray)

    tex_res = cv2.matchTemplate(target_tex, template_tex, cv2.TM_CCOEFF_NORMED)

    color_norm = normalize_01(color_res)
    tex_norm   = normalize_01(tex_res)

    ct_sum = WEIGHT_COLOR + WEIGHT_TEXTURE
    if ct_sum > 1e-6:
        color_tex_map = (
            (WEIGHT_COLOR / ct_sum) * color_norm +
            (WEIGHT_TEXTURE / ct_sum) * tex_norm
        )
    else:
        if log:
            print("[提示] WEIGHT_COLOR 和 WEIGHT_TEXTURE 都为 0，初筛使用平均值。")
        color_tex_map = 0.5 * (color_norm + tex_norm)

    color_tex_map = normalize_01(color_tex_map)

    loc = np.where(color_tex_map >= MATCHING_THRESHOLD)
    boxes = []
    base_scores = []

    for pt_y, pt_x in zip(*loc):
        x1, y1 = int(pt_x), int(pt_y)
        x2, y2 = x1 + tw - 1, y1 + th - 1
        boxes.append([x1, y1, x2, y2])
        base_scores.append(float(color_tex_map[pt_y, pt_x]))

    if log:
        print(f"[INFO] 阈值 = {MATCHING_THRESHOLD:.3f}, 初始候选框数量: {len(boxes)}")

    if not boxes:
        if log:
            print("[WARN] 没有候选框达到阈值。")
        if save_visual or save_txt:
            os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        return []

    keep = non_max_suppression(boxes, base_scores, iou_threshold=MAX_OVERLAP_IOU)
    final_boxes       = [boxes[i]       for i in keep]
    final_base_scores = [base_scores[i] for i in keep]

    if log:
        print(f"[INFO] NMS 后保留框数量: {len(final_boxes)}")

    results = []
    for (x1, y1, x2, y2), base_score in zip(final_boxes, final_base_scores):
        color_sim   = float(color_norm[y1, x1])
        texture_sim = float(tex_norm[y1, x1])

        patch_gray = target_gray[y1:y1+th, x1:x1+tw]
        if patch_gray.shape != template_gray.shape:
            if log:
                print(f"[警告] patch 形状异常，跳过: ({x1}, {y1}, {x2}, {y2})")
            continue

        mse_norm = compute_mse_norm(patch_gray, template_gray)
        mse_sim  = 1.0 - mse_norm
        mse_sim  = float(np.clip(mse_sim, 0.0, 1.0))

        ssim_val = compute_ssim_val(patch_gray, template_gray)
        ssim_sim = (ssim_val + 1.0) / 2.0
        ssim_sim = float(np.clip(ssim_sim, 0.0, 1.0))

        final_score = (
            WEIGHT_COLOR   * color_sim +
            WEIGHT_TEXTURE * texture_sim +
            WEIGHT_MSE     * mse_sim +
            WEIGHT_SSIM    * ssim_sim
        )

        results.append({
            "box": (x1, y1, x2, y2),
            "final_score": final_score,
        })

    results.sort(key=lambda r: r["final_score"], reverse=True)
    boxes_out = [r["box"] for r in results]

    if save_visual or save_txt:
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    if save_visual:
        vis = target.copy()
        for r in results:
            x1, y1, x2, y2 = r["box"]
            score_text = f"{r['final_score']:.2f}"
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 4)
            cv2.putText(
                vis, score_text, (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA
            )
        cv2.imwrite(OUTPUT_PATH, vis)
        if log:
            print(f"[INFO] 结果可视化图已保存到: {OUTPUT_PATH}")

    if save_txt:
        with open(OUTPUT_TXT_PATH, "w", encoding="utf-8") as f:
            f.write(f"Image: {target_path}\n")
            for (x1, y1, x2, y2) in boxes_out:
                f.write(f"Rect: ({x1}, {y1}, {x2}, {y2})\n")
        if log:
            print(f"[INFO] 框坐标已写入: {OUTPUT_TXT_PATH}")

    return boxes_out


# ===============================
# 参数向量 <-> 真实参数 的映射
# ===============================

def _params_to_theta(
    th: float,
    w_color: float,
    w_tex: float,
    w_mse: float,
    w_ssim: float,
) -> np.ndarray:
    """把 (threshold, 四个权重) 映射到 5 维向量 theta。"""
    th = float(np.clip(th, 0.15, 0.85))
    s = (th - 0.15) / 0.7
    s = float(np.clip(s, 1e-5, 1.0 - 1e-5))
    th_logit = math.log(s / (1.0 - s))

    w = np.array([w_color, w_tex, w_mse, w_ssim], dtype=np.float64)
    w = w / (w.sum() + 1e-8)
    w_logits = np.log(w + 1e-8)

    theta = np.zeros(5, dtype=np.float64)
    theta[0] = th_logit
    theta[1:5] = w_logits
    return theta


def _theta_to_params(theta: np.ndarray):
    """把 5 维 theta 还原成 threshold 和四个权重。"""
    assert theta.shape[0] == 5
    th_logit = float(theta[0])
    w_logits = theta[1:5]

    s = 1.0 / (1.0 + math.exp(-th_logit))
    threshold = 0.15 + 0.7 * s

    m = np.max(w_logits)
    e = np.exp(w_logits - m)
    w = e / (e.sum() + 1e-8)
    w_color, w_tex, w_mse, w_ssim = [float(x) for x in w]

    return threshold, w_color, w_tex, w_mse, w_ssim


def _apply_theta(theta: np.ndarray):
    """将 theta 映射成真实参数并写回全局变量。"""
    global MATCHING_THRESHOLD, WEIGHT_COLOR, WEIGHT_TEXTURE, WEIGHT_MSE, WEIGHT_SSIM
    th, wc, wt, wm, ws = _theta_to_params(theta)
    MATCHING_THRESHOLD = th
    WEIGHT_COLOR   = wc
    WEIGHT_TEXTURE = wt
    WEIGHT_MSE     = wm
    WEIGHT_SSIM    = ws


def _evaluate_loss_for_theta(theta: np.ndarray, gt_rects):
    """
    给定 theta:
      1) 应用参数
      2) 跑一次检测（不打印、不保存）
      3) 用 GT 评估
      4) 返回 loss（越小越好） + 指标
    """
    _apply_theta(theta)

    boxes = run_detection(
        template_path=TEMPLATE_PATH,
        target_path=TARGET_PATH,
        save_visual=False,
        save_txt=False,
        log=False,
    )

    correct, wrong, total_detected, total_gt, acc = \
        evaluate_boxes_for_image(boxes, TARGET_PATH, gt_rects)

    fn = max(0, total_gt - correct)  # 漏检数
    lambda_fn = 2.0  # 漏检惩罚系数，可以调整
    loss = wrong + lambda_fn * fn

    return loss, (correct, wrong, total_detected, total_gt, acc)


# ===============================
# SPSA + Adam 训练（带 tqdm）
# ===============================

def train(num_iters: int = 100, return_best: bool = False, show_summary: bool = True):
    """
    使用 SPSA + Adam 对超参数进行黑盒优化。
    只针对当前设置的 TEMPLATE_PATH/TARGET_PATH/GT_PATH。

    训练过程中只显示 tqdm 进度条，不打印其它信息。
    结束后（如果 show_summary=True）打印历史最佳结果和参数。

    return_best=True 时，会额外返回一个 dict：
        {
            "best_loss": float,
            "best_metrics": (correct, wrong, detected, gt, acc),
            "best_theta": np.ndarray(5,)
        }
    """
    if not os.path.exists(GT_PATH):
        raise FileNotFoundError(f"未找到 ground truth 文件: {GT_PATH}")

    gt_rects = parse_gt_file(GT_PATH)

    theta = _params_to_theta(
        MATCHING_THRESHOLD,
        WEIGHT_COLOR,
        WEIGHT_TEXTURE,
        WEIGHT_MSE,
        WEIGHT_SSIM,
    )

    alpha = 0.05
    beta1 = 0.9
    beta2 = 0.999
    eps   = 1e-8

    m = np.zeros_like(theta, dtype=np.float64)
    v = np.zeros_like(theta, dtype=np.float64)

    c0 = 0.1

    best_loss = None
    best_theta = theta.copy()
    best_metrics = None

    for k in trange(1, num_iters + 1, desc="Training", ncols=80):
        ck = c0 / (k ** 0.101)
        delta = np.where(np.random.rand(theta.size) > 0.5, 1.0, -1.0)

        theta_plus  = theta + ck * delta
        theta_minus = theta - ck * delta

        loss_plus,  metrics_plus  = _evaluate_loss_for_theta(theta_plus,  gt_rects)
        loss_minus, metrics_minus = _evaluate_loss_for_theta(theta_minus, gt_rects)

        gk = (loss_plus - loss_minus) / (2.0 * ck * delta)
        gk = gk.astype(np.float64)

        m = beta1 * m + (1.0 - beta1) * gk
        v = beta2 * v + (1.0 - beta2) * (gk * gk)

        m_hat = m / (1.0 - beta1 ** k)
        v_hat = v / (1.0 - beta2 ** k)

        ak = alpha / (k ** 0.101)
        theta = theta - ak * m_hat / (np.sqrt(v_hat) + eps)

        if loss_plus < loss_minus:
            cur_loss = loss_plus
            cur_metrics = metrics_plus
            cur_theta_for_best = theta_plus
        else:
            cur_loss = loss_minus
            cur_metrics = metrics_minus
            cur_theta_for_best = theta_minus

        c, w, td, tg, acc = cur_metrics

        if best_loss is None:
            best_loss = cur_loss
            best_theta = cur_theta_for_best.copy()
            best_metrics = cur_metrics
        else:
            bc, bw, btd, btg, bacc = best_metrics
            if (cur_loss < best_loss) or (abs(cur_loss - best_loss) < 1e-6 and c > bc):
                best_loss = cur_loss
                best_theta = cur_theta_for_best.copy()
                best_metrics = cur_metrics

    # 训练结束，应用最佳参数
    _apply_theta(best_theta)
    c, w, td, tg, acc = best_metrics

    # ✅ 是否打印“历史最佳结果”由 show_summary 控制
    if show_summary:
        print("\n======== Train 结束 ========")
        # print(f"loss    = {best_loss:.4f}")
        # print(f"correct = {c}")
        # print(f"wrong   = {w}")
        # print(f"detected= {td}")
        # print(f"gt      = {tg}")
        # print(f"accuracy= {acc:.4f}")
        print("最佳参数：")
        print(f"  MATCHING_THRESHOLD = {MATCHING_THRESHOLD:.4f}")
        print(f"  WEIGHT_COLOR   = {WEIGHT_COLOR:.4f}")
        print(f"  WEIGHT_TEXTURE = {WEIGHT_TEXTURE:.4f}")
        print(f"  WEIGHT_MSE     = {WEIGHT_MSE:.4f}")
        print(f"  WEIGHT_SSIM    = {WEIGHT_SSIM:.4f}")

    if return_best:
        return {
            "best_loss": float(best_loss),
            "best_metrics": best_metrics,
            "best_theta": best_theta.copy(),
        }


# ===============================
# test: 用当前参数跑一次检测 + 评估
# ===============================

def test(eval_save_path: Optional[str] = None, log: bool = True):
    """
    用当前参数对当前 TARGET_PATH / GT_PATH 做一次 test。

    参数：
        eval_save_path: 若不为 None，则把评估结果写入该 txt 文件。
        log: 是否在控制台打印（默认 True，行为与原来相同）。

    输出：
        - 检测可视化图：OUTPUT_PATH
        - 检测框 txt：OUTPUT_TXT_PATH
        - （可选）评估结果 txt：eval_save_path
    """
    if not os.path.exists(GT_PATH):
        raise FileNotFoundError(f"未找到 ground truth 文件: {GT_PATH}")

    gt_rects = parse_gt_file(GT_PATH)

    if log:
        print("======== 开始 test() ========")
        print(
            f"当前参数: TH={MATCHING_THRESHOLD:.3f}, "
            f"Wc={WEIGHT_COLOR:.3f}, Wt={WEIGHT_TEXTURE:.3f}, "
            f"Wm={WEIGHT_MSE:.3f}, Ws={WEIGHT_SSIM:.3f}"
        )

    boxes = run_detection(
        template_path=TEMPLATE_PATH,
        target_path=TARGET_PATH,
        save_visual=True,
        save_txt=True,
        log=log,
    )

    correct, wrong, total_detected, total_gt, acc = \
        evaluate_boxes_for_image(boxes, TARGET_PATH, gt_rects)

    if log:
        print("\n===== test() 评估结果 =====")
        print(f"Ground truth 中 Rect 总数 (GT): {total_gt}")
        print(f"检测出的总数 (Detected): {total_detected}")
        print(f"匹配正确个数 (True Positive): {correct}")
        print(f"错误个数 (False Positive): {wrong}")
        print(f"正确率 (correct / total_gt): {acc:.4f}")

    if eval_save_path is not None:
        os.makedirs(os.path.dirname(eval_save_path), exist_ok=True)
        with open(eval_save_path, "w", encoding="utf-8") as f:
            f.write(f"Image: {TARGET_PATH}\n")
            f.write(f"GT: {GT_PATH}\n")
            f.write(f"correct: {correct}\n")
            f.write(f"wrong: {wrong}\n")
            f.write(f"detected: {total_detected}\n")
            f.write(f"gt_total: {total_gt}\n")
            f.write(f"accuracy: {acc:.4f}\n")

    # 方便多图脚本使用
    return correct, wrong, total_detected, total_gt, acc


if __name__ == "__main__":
    # 保持原有单图调参+测试功能
    # train(num_iters=300, return_best=False)
    test()
