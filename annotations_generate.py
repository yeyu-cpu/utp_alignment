import cv2
import os

# ============ 可修改部分 ============
IMAGE_PATH = "dataset/with_film_ligted_enhanced/data_film_lighted_enhanced/2026-01-07-18-54-40.jpg"
OUTPUT_TXT = f"dataset/with_film_ligted_enhanced/annotations_gc/annotations_{os.path.basename(IMAGE_PATH)}.txt"

# 显示窗口最大尺寸（你想窗口小就改这里）
MAX_WIN_W, MAX_WIN_H = 800, 500
# ===================================

drawing = False
ix, iy = -1, -1

# 保存所有 (rect_orig, point_orig)
rects_and_points = []

# 当前矩形（原图坐标 & 显示坐标都存）
current_rect_orig = None      # (x1, y1, x2, y2) in original image coords
current_rect_disp = None      # (x1, y1, x2, y2) in display coords

img_orig = cv2.imread(IMAGE_PATH)
if img_orig is None:
    raise FileNotFoundError(f"Cannot read image: {IMAGE_PATH}")

h0, w0 = img_orig.shape[:2]
scale = min(MAX_WIN_W / w0, MAX_WIN_H / h0, 1.0)
disp_w, disp_h = int(w0 * scale), int(h0 * scale)

img_disp = cv2.resize(img_orig, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

# 用于显示（会不断被重画）
img_show = img_disp.copy()
clone = img_disp.copy()  # 固化已标注内容（显示用）

def to_orig(xd, yd):
    """显示坐标 -> 原图坐标"""
    xo = int(round(xd / scale))
    yo = int(round(yd / scale))
    xo = max(0, min(xo, w0 - 1))
    yo = max(0, min(yo, h0 - 1))
    return xo, yo

def norm_rect(x1, y1, x2, y2):
    """规范化矩形，保证左上右下"""
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, img_show, clone, current_rect_orig, current_rect_disp

    # 左键按下：开始画矩形（显示坐标）
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        current_rect_orig = None
        current_rect_disp = None

    # 左键拖动：实时显示矩形（画在显示图上）
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        img_show = clone.copy()
        cv2.rectangle(img_show, (ix, iy), (x, y), (0, 255, 0), 1)

    # 左键抬起：确定矩形（显示坐标 + 原图坐标都记录）
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

        # 显示坐标规范化矩形
        dx1, dy1, dx2, dy2 = norm_rect(ix, iy, x, y)
        current_rect_disp = (dx1, dy1, dx2, dy2)

        # 映射到原图坐标
        ox1, oy1 = to_orig(dx1, dy1)
        ox2, oy2 = to_orig(dx2, dy2)
        ox1, oy1, ox2, oy2 = norm_rect(ox1, oy1, ox2, oy2)
        current_rect_orig = (ox1, oy1, ox2, oy2)

        img_show = clone.copy()
        cv2.rectangle(img_show, (dx1, dy1), (dx2, dy2), (0, 255, 0), 1)

    # 右键点击：标注点（显示坐标画点；保存原图坐标点）
    elif event == cv2.EVENT_RBUTTONDOWN:
        if current_rect_orig is None or current_rect_disp is None:
            print("⚠️ 请先用左键拖动完成一个矩形，再右键点中心点。")
            return

        # 画点在显示图
        img_show = clone.copy()
        # 先把当前框也画出来（避免用户误解）
        dx1, dy1, dx2, dy2 = current_rect_disp
        cv2.rectangle(img_show, (dx1, dy1), (dx2, dy2), (0, 255, 0), 1)
        cv2.circle(img_show, (x, y), 1, (0, 0, 255), -1)

        # 映射点到原图坐标
        px, py = to_orig(x, y)

        rects_and_points.append((current_rect_orig, (px, py)))
        print(f"记录一组：矩形(原图) {current_rect_orig} 点(原图) {(px, py)}")

        # 固化到 clone（显示用）
        clone = img_show.copy()

        # 清空当前框，避免同一矩形被重复点多次（你想允许多个点就删这两行）
        current_rect_orig = None
        current_rect_disp = None

win_name = "annotate: L-drag box, R-click point, ESC save & quit"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win_name, disp_w, disp_h)  # 让窗口贴合显示图大小
cv2.setMouseCallback(win_name, draw_rectangle)

print("🖱️ 操作说明：")
print("  1. 在图片窗口按住左键并拖动来画框，右键点击标注中心点。一次左键然后一次右键为一组数据")
print("  2. 松开后自动裁剪并保存到 OUTPUT_DIR")
print("  3. 按 ESC 键退出程序\n")
print("  P.S.右键点击后，显示的红色中心点在视觉上有偏移，但是实际数据是正确的。标注时只需确保十字光标点击到正确的位置即可")


while True:
    cv2.imshow(win_name, img_show)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cv2.destroyAllWindows()

# ================= 写入 TXT 文件（原图坐标） =================
os.makedirs(os.path.dirname(OUTPUT_TXT), exist_ok=True)

with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
    f.write(f"Image: {IMAGE_PATH}\n")
    for rect, point in rects_and_points:
        x1, y1, x2, y2 = rect
        px, py = point
        f.write(f"Rect: ({x1}, {y1}, {x2}, {y2})  Point: ({px}, {py})\n")

print(f"\n标注完成！已写入到 {OUTPUT_TXT}")
