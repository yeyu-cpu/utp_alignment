#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import Optional, Tuple, List

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


# ================================
# 配置区域（你可以自由修改）
# ================================
PAIRS_LIST_TXT = "dataset/detector_list.txt"   # 每行: 目标图路径 模板图路径
OUTPUT_DIR     = "results/detector_results"  # 所有结果都输出到这个目录下

# 模板匹配参数
MATCHING_THRESHOLD = 0.52   # 初筛阈值，可以按需要微调
WEIGHT_COLOR   = 0.24
WEIGHT_TEXTURE = 0.30
WEIGHT_MSE     = 0.24
WEIGHT_SSIM    = 0.22

MAX_OVERLAP_IOU = 0.4       # NMS 阈值
SOBEL_KSIZE     = 3         # 纹理提取用的 Sobel 核大小

# 圆检测 Hough 参数
HOUGH_PARAM1 = 100
HOUGH_PARAM2_LIST = [30, 25, 20, 18, 16, 14]  # 依次放宽
HOUGH_DP = 1.2
# ================================


# ============ 工具函数（来自原逻辑） ============
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


# ============ 读取多组目标图/模板图路径 ============
def load_pairs(list_txt_path: str) -> List[Tuple[str, str]]:
    """
    从 txt 中读取多组 (target_path, template_path)
    每行格式: 目标图路径 模板图路径
    行首 '#' 视为注释行，会被忽略。
    """
    pairs = []
    with open(list_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                print(f"[WARN] 行格式不正确(需要: 目标图 模板图): {line}")
                continue
            target_path, template_path = parts[0], parts[1]
            pairs.append((target_path, template_path))
    return pairs


# ============ 1. 模板匹配：只取“最优一个矩形框” ============
def detect_best_rectangle(
    template_path: str,
    target_path: str,
    log: bool = True
) -> Optional[Tuple[int, int, int, int]]:
    """
    做颜色+纹理+MSE+SSIM 加权匹配，最后只保留“得分最高的一个矩形框”。
    返回: (x1, y1, x2, y2) 或 None
    """
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
            f"[INFO] 当前模板匹配参数: TH={MATCHING_THRESHOLD:.3f}, "
            f"W_COLOR={WEIGHT_COLOR:.3f}, W_TEXTURE={WEIGHT_TEXTURE:.3f}, "
            f"W_MSE={WEIGHT_MSE:.3f}, W_SSIM={WEIGHT_SSIM:.3f}"
        )

    # --- 颜色匹配 ---
    template_color = template.astype(np.float32)
    target_color   = target.astype(np.float32)
    color_res = cv2.matchTemplate(target_color, template_color, cv2.TM_CCOEFF_NORMED)

    # --- 纹理匹配 ---
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    target_gray   = cv2.cvtColor(target,   cv2.COLOR_BGR2GRAY)
    template_tex  = compute_texture(template_gray)
    target_tex    = compute_texture(target_gray)
    tex_res = cv2.matchTemplate(target_tex, template_tex, cv2.TM_CCOEFF_NORMED)

    # 归一化颜色/纹理得分
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

    # 初筛：找出超过阈值的位置
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
        print("[WARN] 没有候选框达到阈值，无法找到模板所在矩形框。")
        return None

    # NMS 去重
    keep = non_max_suppression(boxes, base_scores, iou_threshold=MAX_OVERLAP_IOU)
    final_boxes       = [boxes[i]       for i in keep]
    final_base_scores = [base_scores[i] for i in keep]

    if log:
        print(f"[INFO] NMS 后保留框数量: {len(final_boxes)}")

    # 对每个框再算完整得分（加入 MSE + SSIM）
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

    if not results:
        print("[WARN] 所有候选框在 MSE/SSIM 检查时都被过滤，无法确定矩形框。")
        return None

    # 只取得分最高的一个
    results.sort(key=lambda r: r["final_score"], reverse=True)
    best_box  = results[0]["box"]
    best_score = results[0]["final_score"]

    if log:
        print(f"[INFO] 最优矩形框: {best_box}, score={best_score:.4f}")

    return best_box


# ============ 2. 在裁剪图上检测圆心（Hough + 自适应半径范围） ============
def detect_circle_hough_in_crop(
    crop_bgr,
    min_radius: Optional[int] = None,
    max_radius: Optional[int] = None
):
    """
    在裁剪图 crop_bgr 上使用 HoughCircles 检测圆。
    没有指定半径范围时，根据裁剪尺寸自动估计：
        min_radius ≈ 0.4 * min(h, w)
        max_radius ≈ 0.8 * min(h, w)
    返回 (cx, cy, r) 或 None（坐标是裁剪图内的）
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 对比度限制自适应直方图均衡
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray_blur)

    h, w = gray_eq.shape[:2]
    center = np.array([w / 2.0, h / 2.0])

    # 自动估计半径范围
    if min_radius is None or max_radius is None:
        min_r = int(0.4 * min(h, w))
        max_r = int(0.8 * min(h, w))
        min_radius = max(min_r, 1)
        max_radius = max(max_r, min_radius + 1)

    for param2 in HOUGH_PARAM2_LIST:
        circles = cv2.HoughCircles(
            gray_eq,
            cv2.HOUGH_GRADIENT,
            dp=HOUGH_DP,
            minDist=min(h, w) / 4.0,
            param1=HOUGH_PARAM1,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius
        )
        if circles is not None:
            circles = np.squeeze(circles, axis=0)
            # 选取最靠近裁剪图中心的圆
            dists = np.linalg.norm(circles[:, :2] - center, axis=1)
            idx = np.argmin(dists)
            x, y, r = circles[idx]
            return float(x), float(y), float(r)

    return None


# ============ 3. 主逻辑：批量处理多组图像 ============
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pairs = load_pairs(PAIRS_LIST_TXT)
    if not pairs:
        print(f"[ERROR] 在 {PAIRS_LIST_TXT} 中没有读取到有效的 (目标, 模板) 路径对。")
        exit(1)

    print(f"[INFO] 共读取到 {len(pairs)} 组图像要处理。")

    for idx, (target_path, template_path) in enumerate(pairs, start=1):
        print("\n" + "=" * 60)
        print(f"[PAIR {idx}] 目标图: {target_path}")
        print(f"[PAIR {idx}] 模板图: {template_path}")

        # 以目标图文件名作为后缀
        target_basename = os.path.splitext(os.path.basename(target_path))[0]
        output_img_path = os.path.join(
            OUTPUT_DIR, f"result_detector_{target_basename}.png"
        )
        output_txt_path = os.path.join(
            OUTPUT_DIR, f"result_detector_circles_{target_basename}.txt"
        )

        try:
            # 1. 模板匹配：找到最优矩形框
            best_box = detect_best_rectangle(template_path, target_path, log=True)
        except FileNotFoundError as e:
            print(f"[ERROR] {e}")
            continue

        if best_box is None:
            print("❌ 未能找到探测器矩形框，本对图像后续圆检测终止。")
            # 如果你希望仍然输出一张“原图”或标记失败的图，也可以在这里写。
            continue

        x1, y1, x2, y2 = best_box
        print(f"[RESULT] 最优矩形框坐标: ({x1}, {y1}, {x2}, {y2})")

        # 2. 读取大图并裁剪 ROI
        target_img = cv2.imread(target_path, cv2.IMREAD_COLOR)
        if target_img is None:
            print(f"[ERROR] 无法读取目标图像: {target_path}")
            continue

        H, W = target_img.shape[:2]
        # 边界保护
        x1 = max(0, min(x1, W - 1))
        x2 = max(0, min(x2, W))
        y1 = max(0, min(y1, H - 1))
        y2 = max(0, min(y2, H))

        crop = target_img[y1:y2, x1:x2].copy()

        # 3. 在裁剪图中检测圆（通光孔）
        circle = detect_circle_hough_in_crop(crop)
        detected_circles = []  # 方便以后扩展为多实例

        vis = target_img.copy()

        if circle is not None:
            cx, cy, r = circle
            cx_i, cy_i, r_i = int(round(cx)), int(round(cy)), int(round(r))

            # 在裁剪图上画圆与圆心（如果需要中间可视化，可以单独保存 crop）
            cv2.circle(crop, (cx_i, cy_i), r_i, (0, 0, 255), 2)
            cv2.circle(crop, (cx_i, cy_i), 3, (0, 255, 0), -1)

            # 映射到整张大图坐标
            global_cx = cx_i + x1
            global_cy = cy_i + y1

            # 在整张图上画矩形框和圆
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)               # 探测器矩形框
            cv2.circle(vis, (global_cx, global_cy), r_i, (0, 0, 255), 2)         # 通光孔圆
            cv2.circle(vis, (global_cx, global_cy), 3, (0, 255, 0), -1)          # 圆心

            cv2.imwrite(output_img_path, vis)

            print(
                f"[RESULT] 通光孔检测成功: center=({global_cx},{global_cy}), "
                f"radius={r_i} (单位: 像素, 坐标为整张大图坐标)"
            )

            detected_circles.append((0, global_cx, global_cy, r_i))
        else:
            print("⚠️ 在矩形框内部未检测到可靠的圆（通光孔）。")
            # 只画矩形框
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imwrite(output_img_path, vis)

        print(f"[INFO] 可视化结果已保存到: {output_img_path}")

        # 4. 写 txt（圆心坐标）
        if detected_circles:
            with open(output_txt_path, "w", encoding="utf-8") as f:
                f.write(f"Image: {target_path}\n")
                f.write(f"Template: {template_path}\n")
                f.write(f"BestRect: ({x1}, {y1}, {x2}, {y2})\n")
                for inst_idx, cx, cy, r in detected_circles:
                    line = f"instance {inst_idx}: center=({cx},{cy}), radius={r}\n"
                    f.write(line)
            print(f"[INFO] 圆心坐标已写入: {output_txt_path}")
        else:
            print("[INFO] 没有圆被检测到，本次不生成圆心坐标 txt 文件。")

    print("\n[INFO] 所有图像处理完成。")
