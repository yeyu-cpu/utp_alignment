import cv2
import os

IMAGE_PATH = r"dataset/with_film_ligted_enhanced/data_film_lighted_enhanced/2026-01-07-18-54-40.jpg"
OUTPUT_DIR = r"dataset/with_film_ligted_enhanced/data_templates_gc"

# ===== 全局 =====
drawing = False
ix, iy = -1, -1

img = None          # 原图（用于裁剪保存）
img_disp = None     # 显示图（缩放后用于显示和画框）
img_show = None     # 当前显示（带框）
scale = 1.0         # 显示图相对原图的缩放比例（disp = orig * scale）


def to_orig(xd, yd):
    """显示坐标 -> 原图坐标"""
    xo = int(xd / scale)
    yo = int(yd / scale)
    # 防止越界
    h, w = img.shape[:2]
    xo = max(0, min(xo, w - 1))
    yo = max(0, min(yo, h - 1))
    return xo, yo


def mouse_callback(event, x, y, flags, param):
    global ix, iy, drawing, img_show

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        img_show = img_disp.copy()
        cv2.rectangle(img_show, (ix, iy), (x, y), (0, 255, 0), 1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        img_show = img_disp.copy()
        cv2.rectangle(img_show, (ix, iy), (x, y), (0, 255, 0), 1)

        # 显示坐标 -> 原图坐标
        ox1, oy1 = to_orig(ix, iy)
        ox2, oy2 = to_orig(x, y)

        x1, x2 = sorted([ox1, ox2])
        y1, y2 = sorted([oy1, oy2])

        if (x2 - x1) < 2 or (y2 - y1) < 2:
            print("⚠️ 选区太小，未执行裁剪。")
            return

        crop = img[y1:y2, x1:x2]

        base_name = os.path.basename(IMAGE_PATH)
        save_name = f"template_{base_name}.png"
        save_path = os.path.join(OUTPUT_DIR, save_name)

        cv2.imwrite(save_path, crop)
        print(f"✅ 已保存裁剪图片：{save_path} （宽{ x2-x1 }px，高{ y2-y1 }px）")


def main():
    global img, img_disp, img_show, scale

    if not os.path.exists(IMAGE_PATH):
        print(f"❌ 找不到图片：{IMAGE_PATH}")
        return
    if not os.path.exists(OUTPUT_DIR):
        print(f"❌ 保存目录不存在：{OUTPUT_DIR}")
        return

    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print("❌ 图片读取失败，请检查文件格式。")
        return

    # === 计算显示缩放比例（让图像适配窗口大小）===
    max_w, max_h = 1000, 600
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)

    disp_w, disp_h = int(w * scale), int(h * scale)
    img_disp = cv2.resize(img, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
    img_show = img_disp.copy()

    win_name = "crop template, left click to drag the temp box, esc to quit"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, disp_w, disp_h)
    cv2.setMouseCallback(win_name, mouse_callback)

    print("🖱️ 操作说明：")
    print("  1. 在图片窗口按住左键并拖动来画框")
    print("  2. 松开后自动裁剪并保存到 OUTPUT_DIR")
    print("  3. 按 ESC 键退出程序\n")

    while True:
        cv2.imshow(win_name, img_show)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
