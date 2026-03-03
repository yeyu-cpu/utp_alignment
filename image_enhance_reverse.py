import re
from pathlib import Path
import cv2
import numpy as np

# ===================== 你需要改的配置 =====================
TXT_PATH = r"results/test_results/test_0000_2023-12-06-11-10-47.jpg.txt"     # 你的坐标文件（只对应一张图）
ORIGINAL_ROOT = r"dataset/data"   # 原图根目录（与 enhanced 同名文件在这里）
OUT_PATH = f"results/test_results/{Path(Path(TXT_PATH).stem).stem}_original.png"       # 输出可视化图片路径
SCALE = 4                                  # 增强时的放大倍率：2 或 4
USE_CENTER_MAPPING = True                  # True 更严谨；False 用 x/scale 简化
SHOW = True                                # 是否弹窗显示（无GUI环境可改 False）
# =========================================================

RECT_PATTERN = re.compile(r"Rect:\s*\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)")
IMG_PATTERN = re.compile(r"Image:\s*(.+)$")

def map_point_up_to_orig(x2, y2, scale, center_mapping=True):
    """增强图坐标 -> 原图连续坐标"""
    if center_mapping:
        # 更贴近 OpenCV resize 的像素中心对齐
        x = (x2 + 0.5) / scale - 0.5
        y = (y2 + 0.5) / scale - 0.5
    else:
        x = x2 / scale
        y = y2 / scale
    return x, y

def map_rect_up_to_orig(rect, scale, center_mapping=True):
    """增强图rect -> 原图rect（int）"""
    x1, y1, x2, y2 = rect
    X1, Y1 = map_point_up_to_orig(x1, y1, scale, center_mapping)
    X2, Y2 = map_point_up_to_orig(x2, y2, scale, center_mapping)

    # 左上 floor，右下 ceil，避免漏边
    X1 = int(np.floor(X1)); Y1 = int(np.floor(Y1))
    X2 = int(np.ceil(X2));  Y2 = int(np.ceil(Y2))
    return X1, Y1, X2, Y2

def clamp_rect(rect, w, h):
    x1, y1, x2, y2 = rect
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return x1, y1, x2, y2

def parse_one_file(txt_path):
    """
    解析你这种格式：
    Image: xxx
    Rect: (...)
    Rect: (...)
    """
    img_path = None
    rects = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m_img = IMG_PATTERN.match(line)
            if m_img:
                img_path = m_img.group(1).strip()
                continue
            m_rect = RECT_PATTERN.match(line)
            if m_rect:
                rects.append(tuple(map(int, m_rect.groups())))
    if img_path is None:
        raise ValueError("未在txt中找到 Image: ... 行")
    if not rects:
        raise ValueError("未在txt中找到任何 Rect: (...) 行")
    return img_path, rects

def resolve_original_path(enhanced_path_str, original_root):
    """
    你的 txt 里是：dataset\\data_enhanced\\xxx.jpg
    这里默认规则：把路径里的 data_enhanced 替换成 data_original
    如果替换失败，则用 original_root + 文件名兜底。
    """
    p = Path(enhanced_path_str)

    s = str(p)
    if "data_enhanced" in s:
        s2 = s.replace("data_enhanced", "data_original")
        orig = Path(s2)
        # 若替换得到的路径仍不存在，用 original_root 兜底
        if orig.exists():
            return orig

    # 兜底：同名文件在 ORIGINAL_ROOT 下
    return Path(original_root) / p.name

def main():
    enhanced_img_path_str, rects_enh = parse_one_file(TXT_PATH)
    orig_path = resolve_original_path(enhanced_img_path_str, ORIGINAL_ROOT)

    img = cv2.imread(str(orig_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取原图：{orig_path}")

    h, w = img.shape[:2]

    rects_orig = []
    for r in rects_enh:
        r0 = map_rect_up_to_orig(r, SCALE, USE_CENTER_MAPPING)
        r0 = clamp_rect(r0, w, h)
        rects_orig.append(r0)

    vis = img.copy()
    for (x1, y1, x2, y2) in rects_orig:
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    ok = cv2.imwrite(OUT_PATH, vis)
    if not ok:
        raise RuntimeError(f"保存失败：{OUT_PATH}")
    print(f"Saved: {OUT_PATH}")
    print(f"Original image: {orig_path}")
    print(f"Rects: {len(rects_orig)} mapped")

    if SHOW:
        win_name = "mapped rects on original"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 1000, 600)
        cv2.imshow(win_name, vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
