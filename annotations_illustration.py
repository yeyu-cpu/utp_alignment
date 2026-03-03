import cv2
import re
import os

# ====== 配置部分 ======
ANNOTATION_TXT = "dataset/with_film_ligted_enhanced/annotations_gc/annotations_2026-01-06-18-24-43.jpg.txt"
WINDOW_W, WINDOW_H = 1000, 600  # 展示窗口最大尺寸（按比例缩放显示）
# =====================

pattern = re.compile(
    r"Rect:\s*\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)\s*Point:\s*\((\d+),\s*(\d+)\)"
)

current_img_path = None
current_img_orig = None  # 原图（画在这里）
current_img_drawn = None # 已绘制标注的原图副本


def resize_keep_ratio(img, max_w, max_h):
    """按最大窗口尺寸等比缩放（不拉伸），返回缩放图与scale"""
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def show_and_wait(img_orig, win_name="Annotations"):
    """显示图片：统一等比缩放后显示"""
    if img_orig is None:
        return

    img_disp, _ = resize_keep_ratio(img_orig, WINDOW_W, WINDOW_H)

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, img_disp.shape[1], img_disp.shape[0])
    cv2.imshow(win_name, img_disp)

    print("按任意键继续，或按 ESC 退出。")
    key = cv2.waitKey(0) & 0xFF
    if key == 27:
        cv2.destroyAllWindows()
        raise SystemExit


with open(ANNOTATION_TXT, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        # 新的一张图片
        if line.startswith("Image:"):
            # 如果上一张存在，先展示
            if current_img_drawn is not None:
                show_and_wait(current_img_drawn, "Annotations")

            # 读取新的图片
            current_img_path = line.split("Image:")[1].strip()
            print(f"\n加载图片: {current_img_path}")

            current_img_orig = cv2.imread(current_img_path)
            if current_img_orig is None:
                print(f"❌ 无法读取图片：{current_img_path}")
                current_img_drawn = None
                continue

            # 每张新图都用一份可绘制副本
            current_img_drawn = current_img_orig.copy()

        # 解析 Rect 和 Point
        elif line.startswith("Rect:") and current_img_drawn is not None:
            m = pattern.search(line)
            if not m:
                print(f"无法解析这一行: {line}")
                continue

            x1, y1, x2, y2, px, py = map(int, m.groups())

            # 画矩形框（绿色）
            cv2.rectangle(current_img_drawn, (x1, y1), (x2, y2), (0, 255, 0), 7)
            # 画点（红色）
            cv2.circle(current_img_drawn, (px, py), 10, (0, 0, 255), -1)

            print(f"绘制：Rect ({x1}, {y1}, {x2}, {y2}) Point ({px}, {py})")

# 文件读完后，把最后一张图展示出来
if current_img_drawn is not None:
    show_and_wait(current_img_drawn, "Annotations")

cv2.destroyAllWindows()
